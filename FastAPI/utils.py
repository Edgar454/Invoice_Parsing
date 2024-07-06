import re
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Function to process an image and to return the OCR
def process_image(image , model , processor, d_type = torch.float32):
    """ Function that takes an image and perform an OCR using the model DonUT via the task document
    parsing
    
    parameters
    __________
    image : a machine readable image of class PIL or numpy"""
    
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    
    image = image.convert('RGB')
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device, dtype = d_type),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    output = processor.token2json(sequence)
    
    return output