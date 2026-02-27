
import json

def convert_for_vertex(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            data = json.loads(line)
            # Match Vertex AI Gemini SFT format
            vertex_item = {
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": data["system_instruction"]}]
                },
                "contents": data["contents"]
            }
            f.write(json.dumps(vertex_item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(lines)} examples to {output_file}")

if __name__ == "__main__":
    convert_for_vertex('tuning_dataset_amparo_v1.jsonl', 'tuning_dataset_vertex.jsonl')
