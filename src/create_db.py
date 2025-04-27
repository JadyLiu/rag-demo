import argparse
import json
import os
import pprint
import time
import boto3
import uuid
from pathlib import Path
from retrying import retry
from botocore.exceptions import ClientError
from opensearchpy import RequestError
from kb_roles import interactive_sleep, KnowledgeBaseRoles, KBInfo


CHUNKING_STRATEGIES = {
    "FIXED_SIZE": {
        "fixedSizeChunkingConfiguration": {
            "maxTokens": 512,
            "overlapPercentage": 20,
        }
    },
    "HIERARCHICAL": {
        "hierarchicalChunkingConfiguration": {
            "levelConfigurations": [{"maxTokens": 1500}, {"maxTokens": 300}],
            "overlapTokens": 60,
        }
    },
    "SEMANTIC": {
        "semanticChunkingConfiguration": {
            "maxTokens": 512,
            "bufferSize": 1,
            "breakpointPercentileThreshold": 95,
        }
    },
}


class NotSupportedRegionException(Exception):
    """
    Thrown when the script is started with a region name that is not supported by the service
    """

    pass


class KnowledgeBaseCreationException(Exception):
    """
    Thrown when the timeout for creating a Bedrock KnowledgeBase expires
    """

    pass


class CreateKB:
    """
    Creates Bedrock KnoweldgeBase
    Args:
        region_name (str): name of the AWS region
        bucket_name (str): name of the S3 bucket that will be used as a data source
        index_name (str): name of the OpenSearch index
        kb_name (str): name of the KnowledgeBase data source
        vector_store_name (str): name of the vector stote
    """

    def __init__(
        self,
        region_name: str,
        bucket_name: str,
        detail_index_name: str,
        summary_index_name: str,
        detail_kb_name: str,
        summary_kb_name: str,
        vector_store_name: str,
    ) -> None:
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.detail_index_name = detail_index_name
        self.summary_index_name = summary_index_name
        self.detail_kb_name = detail_kb_name
        self.summary_kb_name = summary_kb_name
        self.vector_store_name = vector_store_name
        self.printer = pprint.PrettyPrinter(indent=2)
        self.kb_roles = KnowledgeBaseRoles(region_name)
        #  Create separate KBInfo instances for detail and summary KBs
        self.detail_kb_info = KBInfo(
            index_name=self.detail_index_name,
            bucket_name=self.bucket_name,
            region_name=self.region_name,
            bedrock_execution_role_name=self.kb_roles.bedrock_execution_role_name,
            fm_policy_name=self.kb_roles.fm_policy_name,
            s3_policy_name=self.kb_roles.s3_policy_name,
            oss_policy_name=self.kb_roles.oss_policy_name,
        )
        
        self.summary_kb_info = KBInfo(
            index_name=self.summary_index_name,
            bucket_name=self.bucket_name,
            region_name=self.region_name,
            bedrock_execution_role_name=self.kb_roles.bedrock_execution_role_name,
            fm_policy_name=self.kb_roles.fm_policy_name,
            s3_policy_name=self.kb_roles.s3_policy_name,
            oss_policy_name=self.kb_roles.oss_policy_name,
        )

    def create_bucket(self, s3_client: boto3.client) -> None:
        """
        Creates an S3 bucket if not existing

        Args:
            s3_client (boto3.client): The boto3 client for S3 service.
        """
        try:
            s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} exists")
        except ClientError:
            print(f"Creating bucket {self.bucket_name}")
            s3_client.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={"LocationConstraint": self.region_name},
            )

    def upload_directory(self, path: str) -> None:
        """
        Upload files from the specific directory structure:
        - data/
            - pdf/          # Contains PDFs and metadata JSONs
            - summary-PDFs/ # Contains summary JSONs
        """
        s3_client = boto3.client("s3")
        # Walk through all subdirectories in data/
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, path)
                
                # Determine the correct S3 prefix based on the source folder
                if "summary-PDFs" in relative_path:
                    s3_key = f"summary-PDFs/{os.path.basename(file)}"
                elif "pdf" in relative_path:
                    s3_key = f"pdf/{os.path.basename(file)}"
                else:
                    continue  # Skip files not in pdf or summary-PDFs folders
                    
                print(f"Uploading {file_path} to S3://{self.bucket_name}/{s3_key}")
                try:
                    s3_client.upload_file(
                        file_path,
                        self.bucket_name,
                        s3_key
                    )
                    print(f"Successfully uploaded {s3_key}")
                except Exception as e:
                    print(f"Error uploading {s3_key}: {str(e)}")

    def create_os_polices_and_collection(
        self,
        aoss_client: boto3.client,
    ) -> tuple[dict, str, str]:
        """
        Create security, network, and data access policies within OpenSearch Serverless (OSS),
        and create an OSS collection for the vector store.

        Args:
            aoss_client (boto3.client): The boto3 client for OpenSearch Serverless.

        Returns:
            tuple[dict, str, str]: A tuple containing the created collection, collection ID, and Bedrock execution role ARN.
        """
        bedrock_kb_execution_role = self.kb_roles.create_bedrock_execution_role(
            bucket_name=self.bucket_name
        )
        bedrock_kb_execution_role_arn = bedrock_kb_execution_role["Role"]["Arn"]
        encryption_policy, network_policy, access_policy = (
            self.kb_roles.create_policies_in_oss(
                vector_store_name=self.vector_store_name,
                aoss_client=aoss_client,
                bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn,
            )
        )
        # Update both detail and summary KB info
        self.detail_kb_info.access_policy_name = access_policy["accessPolicyDetail"]["name"]
        self.detail_kb_info.network_policy_name = network_policy["securityPolicyDetail"]["name"]
        self.detail_kb_info.encryption_policy_name = encryption_policy["securityPolicyDetail"]["name"]
        
        self.summary_kb_info.access_policy_name = access_policy["accessPolicyDetail"]["name"]
        self.summary_kb_info.network_policy_name = network_policy["securityPolicyDetail"]["name"]
        self.summary_kb_info.encryption_policy_name = encryption_policy["securityPolicyDetail"]["name"]

        collection = aoss_client.create_collection(
            name=self.vector_store_name, type="VECTORSEARCH"
        )
        self.printer.pprint(collection)

        collection_id = collection["createCollectionDetail"]["id"]
        self.detail_kb_info.collection_id = collection_id
        self.summary_kb_info.collection_id = collection_id
        # wait for collection creation
        # This can take couple of minutes to finish
        response = aoss_client.batch_get_collection(names=[self.vector_store_name])
        # Periodically check collection status
        while (response["collectionDetails"][0]["status"]) == "CREATING":
            print("Creating collection...")
            interactive_sleep(60)
            response = aoss_client.batch_get_collection(names=[self.vector_store_name])
        print("\nCollection successfully created:")
        self.printer.pprint(response["collectionDetails"])

        # create opensearch serverless access policy and attach it to Bedrock execution role
        self.kb_roles.create_oss_policy_attach_bedrock_execution_role(
            collection_id=collection_id,
            bedrock_kb_execution_role=bedrock_kb_execution_role,
        )
        # It can take up to a minute for data access rules to be enforced
        interactive_sleep(90)
        return collection, collection_id, bedrock_kb_execution_role_arn

    def create_vector_index(
        self,
        collection_id: str,
        is_summary: bool = False,
    ) -> None:
        """
        Create a vector index in OpenSearch Serverless with the knn_vector field index mapping,
        specifying the dimension size, name, and engine.

        Args:
            collection_id (str): The ID of the OpenSearch Serverless collection.
            is_summary (bool): If True, creates summary index. If False, creates detail index.
        """
        oss_client = self.kb_roles.create_os_client(collection_id)
        
        # Select the appropriate index name
        index_name = self.summary_index_name if is_summary else self.detail_index_name

        body_json = {
            "settings": {
                "index.knn": "true",
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "innerproduct",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        },
                    },
                    },
                    "text": {"type": "text"},
                    "text-metadata": {"type": "text"},
                }
            },
        }

        try:
            print(f"\nCreating {'summary' if is_summary else 'detail'} index: {index_name}")
            response = oss_client.indices.create(
                index=index_name,
                body=json.dumps(body_json),
                params={"timeout": 300},
            )
            print("\nCreating index:")
            self.printer.pprint(response)

            # index creation can take up to a minute
            interactive_sleep(60)
        except RequestError as e:
            # you can delete the index if its already exists
            # oss_client.indices.delete(index=index_name)
            print(f"Error while trying to create the index {index_name}, with error {e.error}\n")

    def create_knowledge_base(
        self,
        collection: dict,
        bedrock_kb_execution_role_arn: str,
        bedrock_agent_client: boto3.client,
        chunking_strategy: str,
        is_summary: bool = False,
    ) -> tuple[dict, dict]:
        """
        Create a Knowledge Base and a Data Source within the Knowledge Base.

        Args:
            collection (dict): The created OpenSearch Serverless collection.
            bedrock_kb_execution_role_arn (str): The ARN of the Bedrock execution role.
            bedrock_agent_client (boto3.client): The boto3 client for Bedrock Agent.
            chunking_strategy (str): The chunking strategy to use.
            is_summary (bool): If True, creates summary KB. If False, creates detail KB.

        Returns:
            tuple[dict, dict]: A tuple containing the created Knowledge Base and Data Source.
        """

        @retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
        def create_knowledge_base_func():
            create_kb_response = bedrock_agent_client.create_knowledge_base(
                name=self.summary_kb_name if is_summary else self.detail_kb_name,
                roleArn=role_arn,
                knowledgeBaseConfiguration={
                    "type": "VECTOR",
                    "vectorKnowledgeBaseConfiguration": {
                        "embeddingModelArn": embedding_model_arn
                    },
                },
                storageConfiguration={
                    "type": "OPENSEARCH_SERVERLESS",
                    "opensearchServerlessConfiguration": opensearch_serverless_configuration,
                },
            )
            return create_kb_response["knowledgeBase"]

        opensearch_serverless_configuration = {
            "collectionArn": collection["createCollectionDetail"]["arn"],
            "vectorIndexName": self.summary_index_name if is_summary else self.detail_index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata",
            },
        }

        # Ingest strategy - How to ingest data from the data source
        chunking_strategy_configuration = {
            "chunkingStrategy": chunking_strategy,
            **CHUNKING_STRATEGIES[chunking_strategy],
        }

        # Configure S3 data source with appropriate prefix
        s3_configuration = {
            "bucketArn": f"arn:aws:s3:::{self.bucket_name}",
            "inclusionPrefixes": ["summary-PDFs/"] if is_summary else ["pdf/"]
        }

        # The embedding model used by Bedrock to embed ingested documents, and realtime prompts
        embedding_model_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/amazon.titan-embed-text-v2:0"
        role_arn = bedrock_kb_execution_role_arn

        knowledge_base = create_knowledge_base_func()
        self.printer.pprint(knowledge_base)

        kb_name = self.summary_kb_name if is_summary else self.detail_kb_name
        # Create a DataSource in KnowledgeBase
        create_ds_response = bedrock_agent_client.create_data_source(
            name=kb_name,
            knowledgeBaseId=knowledge_base["knowledgeBaseId"],
            dataSourceConfiguration={"type": "S3", "s3Configuration": s3_configuration},
            vectorIngestionConfiguration={
                "chunkingConfiguration": chunking_strategy_configuration
            },
        )
        data_source = create_ds_response["dataSource"]
        self.printer.pprint(data_source)
        bedrock_agent_client.get_data_source(
            knowledgeBaseId=knowledge_base["knowledgeBaseId"],
            dataSourceId=data_source["dataSourceId"],
        )

        kb_response = bedrock_agent_client.get_knowledge_base(
            knowledgeBaseId=knowledge_base["knowledgeBaseId"]
        )
        retries = 0
        while kb_response["knowledgeBase"]["status"] == "CREATING":
            interactive_sleep(30)
            kb_response = bedrock_agent_client.get_knowledge_base(
                knowledgeBaseId=knowledge_base["knowledgeBaseId"]
            )
            retries += 1
            if retries > 20:
                raise KnowledgeBaseCreationException("Failed to create knowledge base")

        # Store KB info in appropriate instance
        if is_summary:
            self.summary_kb_info.kb_id = knowledge_base["knowledgeBaseId"]
            self.summary_kb_info.ds_id = data_source["dataSourceId"]
        else:
            self.detail_kb_info.kb_id = knowledge_base["knowledgeBaseId"]
            self.detail_kb_info.ds_id = data_source["dataSourceId"]

        return knowledge_base, data_source

    def start_ingestion_job(
        self, bedrock_agent_client: boto3.client, kb: dict, ds: dict,  is_summary: bool = False
    ) -> None:
        """
        Start an ingestion job for the created Data Source within the Knowledge Base.

        Args:
            bedrock_agent_client (boto3.client): The boto3 client for Bedrock Agent.
            kb (dict): The created Knowledge Base.
            ds (dict): The created Data Source.
        """
        kb_type = "Summary" if is_summary else "Detail"
        print(f"\nStarting ingestion job for {kb_type} Knowledge Base...")

        start_job_response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=kb["knowledgeBaseId"], dataSourceId=ds["dataSourceId"]
        )
        job = start_job_response["ingestionJob"]
        self.printer.pprint(job)

        while job["status"] in ["IN_PROGRESS", "STARTING"]:
            get_job_response = bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=kb["knowledgeBaseId"],
                dataSourceId=ds["dataSourceId"],
                ingestionJobId=job["ingestionJobId"],
            )
            job = get_job_response["ingestionJob"]
            time.sleep(5)
        self.printer.pprint(job)
        interactive_sleep(40)

        # Print the knowledge base Id in bedrock, that corresponds to the Opensearch index in the collection we created before, we will use it for the invocation later
        kb_id = kb["knowledgeBaseId"]
        self.printer.pprint(kb_id)


def main():
    suffix = uuid.uuid4().hex[:6]
    parser = argparse.ArgumentParser(
        description="Create and ingest documents into two knowledge bases (detail and summary), by default files will be copied from ../data directory to the provided S3 location"
    )
    parser.add_argument(
        "--region_name",
        type=str,
        required=False,
        help="AWS region name",
        default="us-west-2",
    )
    parser.add_argument(
        "--detail_kb_name", type=str, required=False, help="Detail Knowledge base name", default="anycompany-kb-detail"
    )
    parser.add_argument(
        "--summary_kb_name", type=str, required=False, help="Summary Knowledge base name", default="anycompany-kb-summary"
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        required=False,
        help="S3 Bucket name that should save the data",
    )
    parser.add_argument(
        "--use_s3",
        required=False,
        help="If set, we use the files in the provided S3 location (in --bucket_name) without copying ../data location",
        default=False,
    )
    parser.add_argument(
        "--vectorstore_name",
        type=str,
        required=False,
        help="Name of the vector store",
        default=f"demo-rag-vs-{suffix}",
    )
    parser.add_argument(
        "--detail_index_name",
        type=str,
        required=False,
        help="Name of the detail opensearch index",
        default=f"demo-rag-detail-index-{suffix}",
    )
    parser.add_argument(
        "--summary_index_name",
        type=str,
        required=False,
        help="Name of the summary opensearch index",
        default=f"demo-rag-summary-index-{suffix}",
    )
    parser.add_argument(
        "--chunking_strategy",
        type=str,
        required=False,
        help=f"Chunking strategy, choice of {CHUNKING_STRATEGIES.keys()}",
        default="FIXED_SIZE",
    )

    args = parser.parse_args()

    region_name = args.region_name
    allowed_regions = [
        "us-east-1",
        "us-west-2",
        "ap-south-1",
        "ap-southeast-2",
        "ca-central-1",
        "eu-central-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
    ]
    if region_name not in allowed_regions:
        raise NotSupportedRegionException(
            f"The region for needs to be set to one of {allowed_regions}"
        )
    if args.chunking_strategy not in CHUNKING_STRATEGIES.keys():
        raise Exception("Not supported chunking strategy")
    
    boto3.setup_default_session(region_name=region_name)
    boto3_session = boto3.session.Session(region_name=region_name)
    bedrock_agent_client = boto3_session.client(
        "bedrock-agent", region_name=region_name
    )
    s3_client = boto3_session.client("s3")
    aoss_client = boto3_session.client("opensearchserverless")
    s3_suffix = f"{region_name}-{boto3.client('sts').get_caller_identity()['Account']}"
    bucket_name = (
        f"bedrock-kb-{s3_suffix}" if not args.bucket_name else args.bucket_name
    )
    kb_instance = CreateKB(
        region_name,
        bucket_name,
        args.detail_index_name,
        args.summary_index_name,
        args.detail_kb_name,
        args.summary_kb_name,
        args.vectorstore_name,
    )

    # Step 1: Create an S3 bucket if not existing
    kb_instance.create_bucket(s3_client)

    # Step 2: Create OSS policies and collection
    collection, collection_id, bedrock_kb_execution_role_arn = (
        kb_instance.create_os_polices_and_collection(aoss_client)
    )

    # Step 3: Create vector indices
    kb_instance.create_vector_index(collection_id, is_summary=False)
    kb_instance.create_vector_index(collection_id, is_summary=True)

    # Upload data to s3 to the bucket that was configured as a data source to the knowledge base
    if not args.use_s3:
        kb_instance.upload_directory(Path(__file__).parent.absolute() / "data")

    # Step 4: Create Knowledge Bases
    detail_kb, detail_ds = kb_instance.create_knowledge_base(
        collection,
        bedrock_kb_execution_role_arn,
        bedrock_agent_client,
        args.chunking_strategy,
        is_summary=False
    )

    summary_kb, summary_ds = kb_instance.create_knowledge_base(
        collection,
        bedrock_kb_execution_role_arn,
        bedrock_agent_client,
        args.chunking_strategy,
        is_summary=True
    )

    # Step 5: Start ingestion jobs
    kb_instance.start_ingestion_job(bedrock_agent_client, detail_kb, detail_ds, is_summary=False)
    kb_instance.start_ingestion_job(bedrock_agent_client, summary_kb, summary_ds, is_summary=True)

    # Save KB info
    path = Path(__file__).parent.absolute()
    with open(path / f"{args.detail_kb_name}_detail.json", "w", encoding="utf-8") as file:
        json.dump(kb_instance.detail_kb_info.model_dump(), file, indent=4)
    
    with open(path / f"{args.summary_kb_name}_summary.json", "w", encoding="utf-8") as file:
        json.dump(kb_instance.summary_kb_info.model_dump(), file, indent=4)


if __name__ == "__main__":
    main()
