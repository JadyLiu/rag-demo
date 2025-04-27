import os
import json
import PyPDF2
import re
import traceback
import boto3


class PDFProcessor:
    def __init__(self):
        self.session = boto3.Session()
        self.bedrock_runtime = self.session.client(service_name='bedrock-runtime')
        self.input_folder = './data/pdf/'
        self.summary_folder = './data/summary-PDFs/'
        self.model_id = os.environ.get('BEDROCK_MODEL_ID', 'mistral.mistral-large-2407-v1:0')

    def process_pdfs(self):
        os.makedirs(self.summary_folder, exist_ok=True)
        
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.pdf'):
                self._process_single_pdf(filename)

    def _process_single_pdf(self, filename):
        file_path = os.path.join(self.input_folder, filename)
        try:
            document_text = self._extract_text_from_pdf(file_path)
            if document_text is None:
                print(f"Skipping file {filename} due to text extraction error")
                return

            json_response = self._query_bedrock(document_text, filename)
            if json_response:
                self._save_metadata(filename, json_response)
                self._save_full_summary(filename, json_response)
            else:
                print(f'Failed to generate metadata for {filename}')
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            traceback.print_exc()

    @staticmethod
    def _extract_text_from_pdf(file_path):
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + ' '
            return PDFProcessor._clean_text(text)
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
            return None

    @staticmethod
    def _clean_text(text):
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        text = re.sub(r'[\ud800-\udfff]', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def _clean_json_string(json_string):
        json_string = json_string.strip()
        if not json_string.startswith('{'):
            json_string = '{' + json_string
        if not json_string.endswith('}'):
            json_string = json_string + '}'
        json_string = ''.join(char for char in json_string if ord(char) >= 32 or char in '\n\r\t')
        return json_string

    def _query_bedrock(self, text, filename):
        messages = [
            {
                "role": "user",
                "content": f"""Based on the following document content and filename, please extract the title and generate a comprehensive summary of the document. 
                Return the information in JSON format as shown below. Ensure the JSON is complete and valid, starting with an opening curly brace and ending with a closing curly brace:
                {{
                    "metadataAttributes": {{ 
                        "filename": string,
                        "title": string,
                        "summary": string
                    }}
                }}

                Filename: {filename}
                Document content:
                {text}"""
            },
        ]

        body = json.dumps({
            "stop": [],
            "max_tokens": 1000,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 1,
        })

        try:
            response = self.bedrock_runtime.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )

            response_body = json.loads(response.get('body').read())
            print("Raw response:", response_body)  # Debug print
            
            # json_string = response_body['content'][0]['text']
            # Extract the content from the Mistral AI response
            json_string = response_body['choices'][0]['message']['content']
            
            # Remove the markdown code block indicators if present
            json_string = json_string.strip('`')
            if json_string.startswith('json\n'):
                json_string = json_string[5:]  # Remove 'json\n'

            cleaned_json_string = self._clean_json_string(json_string)
            return json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error for file {filename}: {str(e)}")
            print("Problematic JSON string:", cleaned_json_string)
            return None
        except Exception as e:
            print(f"Error processing file {filename} in query_bedrock: {str(e)}")
            traceback.print_exc()
            return None

    def _save_metadata(self, filename, json_response):
        output_filename = os.path.splitext(filename)[0] + '.pdf.metadata.json'
        metadata = {
            "metadataAttributes": {
                "filename": json_response["metadataAttributes"]["filename"],
                "title": json_response["metadataAttributes"]["title"]
            }
        }
        with open(os.path.join(self.input_folder, output_filename), 'w', encoding='utf-8') as jsonfile:
            json.dump(metadata, jsonfile, ensure_ascii=False, indent=4)
        print(f'Metadata file {output_filename} saved in original folder.')

    def _save_full_summary(self, filename, json_response):
        output_filename = os.path.splitext(filename)[0] + '.json'
        with open(os.path.join(self.summary_folder, output_filename), 'w', encoding='utf-8') as jsonfile:
            json.dump(json_response, jsonfile, ensure_ascii=False, indent=4)
        print(f'Full summary file {output_filename} saved in summary-PDFs folder.')


if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_pdfs()
    print("Process completed.")
