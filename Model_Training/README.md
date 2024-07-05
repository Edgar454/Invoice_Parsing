# Training and Optimization

To handle the parsing we used the donut model available via huggingface . We began by training the model with different image size to see the result and after that we applied knowledge distillation and bfloat16 quantization for the lowest size in order to speed up inference and we pushed the model to huggingface hub 

