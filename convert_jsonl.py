
import json

def process_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Parse each line as JSON
            data = json.loads(line.strip())

            # Retrieve necessary fields
            global_idx = data["global_idx"]
            source_img = data["source_img"]
            caption = data["caption"]
            sam_objects = data["sam_objects"]

            # Loop through sam_objects to create the new JSON objects
            obj_num = 0
            for sam_obj in sam_objects:
                # Generate the segment mask path
                segment_mask_path = source_img.replace("ori.jpg", f"mask_{sam_obj}.jpg")
                
                # Construct the new JSON object
                new_entry = {
                    "global_idx": global_idx,
                    "source_image": source_img,
                    "image_path": segment_mask_path,
                    "objects": sam_obj,
                    "segment_idx": obj_num,
                    "input_text": f"The image 0 is <image>\n{caption}"
                }
                obj_num+=1
                # Write the new JSON object to the output file
                outfile.write(json.dumps(new_entry) + "\n")

if __name__ == "__main__":
    # process_data("/nobackup/zefan/projects/VLGen/segment_results_imgnet/total.jsonl", "/nobackup/zefan/projects/VLGen/training_imagenet.jsonl")
    process_data("/nobackup/zefan/projects/VLGen/segment_results_sam/total.jsonl", "/nobackup/zefan/projects/VLGen/training_sam.jsonl")
    
import json  

def segregate_data(input_file1, input_file2, val_range, train_output, val_output):  
    train_entries = []  
    val_entries = []  

    def process_file(input_file):  
        with open(input_file, 'r') as infile:  
            for line in infile:  
                data = json.loads(line.strip())  
                global_idx = data["global_idx"]  
                
                # Segregate based on val_range  
                if val_range[0] <= global_idx <= val_range[1]:  
                    val_entries.append(data)  
                else:  
                    train_entries.append(data)  

    # Process both JSONL files  
    process_file(input_file1)  
    process_file(input_file2)  

    # Save the training entries  
    with open(train_output, 'w') as train_file:  
        for entry in train_entries:  
            train_file.write(json.dumps(entry) + "\n")  

    # Save the validation entries  
    with open(val_output, 'w') as val_file:  
        for entry in val_entries:  
            val_file.write(json.dumps(entry) + "\n")  

if __name__ == "__main__":  
    val_range = (200, 500)  # Define the range for the validation set  
    segregate_data(  
        "/nobackup/zefan/projects/VLGen/training_imagenet.jsonl",  
        "/nobackup/zefan/projects/VLGen/training_sam.jsonl",  
        val_range,  
        "training_set.jsonl",  
        "validation_set.jsonl"  
    )