# 主要用于训练风格、作画能力（需要每张图片都有对应的标签描述�?
export MODEL_NAME="./model"
export INSTANCE_DIR="/home/sunpeiwen/BP4D-256-crop/pics"
export OUTPUT_DIR="/mnt/tempDisk/model/au_nlp_model"
export LOG_DIR="./logs"
export TEST_PROMPTS_FILE="./test_prompts_style.txt"

# rm -rf $LOG_DIR/*
# rm -r /mnt/tempDisk/BP4D-256-crop/enhanced-au-nl-label
# mkdir /mnt/tempDisk/BP4D-256-crop/enhanced-au-nl-label
# python /mnt/tempDisk/BP4D-256-crop/au2label.py

  # WANDB__SERVICE_WAIT=300 \
accelerate launch tools/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --mixed_precision="fp16" \
  --instance_data_dir=$INSTANCE_DIR \
  --use_txt_as_label \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOG_DIR \
  --resolution=128 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --save_model_every_n_steps=1000 \
  --auto_test_model \
  --test_prompts_file=$TEST_PROMPTS_FILE \
  --test_seed=123 \
  --test_num_per_prompt=5 \
  # --train_text_encoder
  
# 如果max_train_steps改大了，请记得把save_model_every_n_steps也改大，不然磁盘容易中间就满�?
# 以下是核心参数介绍：
# 主要的几�?# --train_text_encoder 训练文本编码�?# --mixed_precision="fp16" 混合精度训练
# - center_crop 
# 是否裁剪图片，一般如果你的数据集不是正方形的话，需要裁�?# - resolution 
# 图片的分辨率，一般是512，使用该参数会自动缩放输入图�?# 可以配合center_crop使用，达到裁剪成正方形并缩放�?12*512的效�?# - instance_prompt
# 如果你希望训练的是特定的人物，使用该参数
# �?--instance_prompt="a photo of <xxx> girl"
# - use_txt_as_label
# 是否读取与图片同名的txt文件作为label
# 如果你要训练的是整个大模型的图像风格，那么可以使用该参数
# 该选项会忽略instance_prompt参数传入的内�?# - learning_rate
# 学习率，一般是2e-6，是训练中需要调整的关键参数
# 太大会导致模型不收敛，太小的话，训练速度会变�?# - max_train_steps
# 训练的最大步数，一般是1000，如果你的数据集比较大，那么可以适当增大该�?# - save_model_every_n_steps
# 每多少步保存一次模型，方便查看中间训练的结果找出最优的模型，也可以用于断点续训

# --train_text_encoder # 除了图像生成器，也训练文本编码器

# --auto_test_model, --test_prompts_file, --test_seed, --test_num_per_prompt
# 分别是自动测试模型（每save_model_every_n_steps步后）、测试的文本、随机种子、每个文本测试的次数
# 测试的样本图片会保存在模型输出目录下的test文件夹中
