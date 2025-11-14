import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path


class OpenAPIParser:
    
    def __init__(self, spec: Dict[str, Any], toolkit_name: str):
        self.spec = spec
        self.toolkit_name = toolkit_name
        self.components = spec.get('components', {})
        
    def resolve_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        if not ref.startswith('#/'):
            return None
            
        parts = ref[2:].split('/')
        current = self.spec
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
                
        return current
    
    def extract_schema_properties(self, schema: Dict[str, Any], required_fields: List[str] = None) -> Dict[str, Dict[str, Any]]:
        if required_fields is None:
            required_fields = []
            
        properties = {}
        
        if '$ref' in schema:
            schema = self.resolve_ref(schema['$ref']) or schema
            
        if 'required' in schema:
            required_fields = schema.get('required', [])
            
        schema_properties = schema.get('properties', {})
        for prop_name, prop_schema in schema_properties.items():
            if '$ref' in prop_schema:
                prop_schema = self.resolve_ref(prop_schema['$ref']) or prop_schema
            
            prop_type = prop_schema.get('type', 'string')
            is_required = prop_name in required_fields
            
            properties[prop_name] = {
                'type': prop_type,
                'required': is_required
            }
            
        return properties
    
    def extract_parameters(self, operation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        parameters = {}
        
        for param in operation.get('parameters', []):
            if '$ref' in param:
                param = self.resolve_ref(param['$ref']) or param
                
            param_name = param.get('name', '')
            param_schema = param.get('schema', {})
            
            if '$ref' in param_schema:
                param_schema = self.resolve_ref(param_schema['$ref']) or param_schema
            
            param_type = param_schema.get('type', 'string')
            is_required = param.get('required', False)
            
            parameters[param_name] = {
                'type': param_type,
                'required': is_required
            }
        
        request_body = operation.get('requestBody', {})
        if request_body:
            if '$ref' in request_body:
                request_body = self.resolve_ref(request_body['$ref']) or request_body
            
            content = request_body.get('content', {})
            is_required = request_body.get('required', False)
            
            for content_type in ['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data']:
                if content_type in content:
                    schema = content[content_type].get('schema', {})
                    body_props = self.extract_schema_properties(schema)
                    
                    for prop_name, prop_info in body_props.items():
                        prop_info['required'] = is_required and prop_info['required']
                        parameters[prop_name] = prop_info
                    break
        
        return parameters
    
    def extract_responses(self, operation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        response_params = {}
        
        responses = operation.get('responses', {})
        
        for status_code, response in responses.items():
            if '$ref' in response:
                response = self.resolve_ref(response['$ref']) or response
            
            content = response.get('content', {})
            
            if 'application/json' in content:
                schema = content['application/json'].get('schema', {})
                props = self.extract_schema_properties(schema)
                
                for prop_name, prop_info in props.items():
                    if prop_name not in response_params:
                        response_params[prop_name] = prop_info
        
        return response_params
    
    def convert_to_tool_format(self, param_dict: Dict[str, Dict[str, Any]]) -> str:
        formatted = {}
        for name, info in param_dict.items():
            formatted[name] = {
                'type': info['type'],
                'required': info['required']
            }
        return json.dumps(formatted)
    
    def parse(self) -> List[Dict[str, Any]]:
        tools = []
        
        paths = self.spec.get('paths', {})
        
        for path, path_item in paths.items():
            if '$ref' in path_item:
                path_item = self.resolve_ref(path_item['$ref']) or path_item
            
            for method in ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']:
                if method not in path_item:
                    continue
                
                operation = path_item[method]
                
                operation_id = operation.get('operationId', f"{method}_{path}")
                summary = operation.get('summary', '')
                description = operation.get('description', '')
                tool_description = summary or description or f"{method.upper()} {path}"
                
                input_params = self.extract_parameters(operation)
                response_params = self.extract_responses(operation)
                
                tool = {
                    'toolkit_name': self.toolkit_name,
                    'tool_id': operation_id,
                    'tool_description': tool_description,
                    'tool_input_parameters': self.convert_to_tool_format(input_params),
                    'tool_response_parameters': self.convert_to_tool_format(response_params)
                }
                
                tools.append(tool)
        
        return tools


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenAPI specification to tool definition format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Path to input OpenAPI JSON file'
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='Path to output tool definitions JSON file'
    )
    parser.add_argument(
        '--toolkit-name',
        '-t',
        required=True,
        help='Name for the toolkit (e.g., "googlecalendar", "sendgrid")'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    with open(input_path, 'r') as f:
        openapi_spec = json.load(f)
    
    parser = OpenAPIParser(openapi_spec, args.toolkit_name)
    tools = parser.parse()
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(tools, f, indent=2)
    
    print(f"Successfully converted {len(tools)} operations")
    print(f"Output written to: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
