# MACOE: A Multi-Agent Communication and Order Execution Framework with Dual-Layer Reinforcement Learning
### (under review)

### 1. **Get Data**

    We get data from Qlib, which is collected from Yahoo Finance.
    - Step-1: ```git clone https://github.com/microsoft/qlib.git```, ```pip install pyqlib```: install qlib  
    - Step-2: ```python scripts/get_data.py qlib_data --target_dir ~./cn_1min --region cn```: Get China 1min data

### 2. **Pre-Process**

    We preprocessed the data using Qlib’s pipeline and, for simplicity, consolidated the steps in the ./data_preprocess directory.
    - Step-1: ```python data_preprocess/normalize_feature.py```: get normalized data: ./cn_1min/normed_feature/
    - Step-2: ```python data_preprocess/backtest.py```: get backtest data: ./cn_1min/backtest/
    - Step-3: ```python data_preprocess/order_gen.py```,  ```python order_gen_roll.py```: divide data into three windows

### 3. **Get Test Data**

    To reduce runtime, we pre-generate and store the test dataset before executing the method.
    - Step-1: ```python common/utils_ma_ind_test_gen.py```

### Method

1. **Pre-train:PPO(baseline)**

    - Step-1: ```python pre_train.py```: Get pre-trained model (./model_pretrain/)

2. **Train:HARMO**

    Step-1: ```python main.py```
    ```runner_am_ind_mla_gru.py```: Runner function
    ```maddpg/train_fun_am_mla_gru.py```: Trainer function
    ```agent_am_mla_gru.py```: Critic function
    ```actor_critic_gru.py``: MLMS
    ```actor_critic_am_mla_gru.py```: Agent 

### Requirements
- Python == 3.10.0
- PyTorch == 2.1.1

### Demo Data
    We also provide the demo data to verify runable code, including only 5 trading days of China Market, US Market and CEA. Please change your own path in ```./data_preprocess``` to run the demo code. Follow the Pre-Process above and just change Step-3. Notably, change the batch_size=2 due to the demo data's length.
    Sttp-3: ```python data_preprocess/order_gen.py```,  ```python order_gen_roll_demo.py```

### Sector stock id
    ```ind_stock```: China A-share
    ```us_stock```: US S&P 500

