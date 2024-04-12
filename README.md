# LLaVA_ChatBot
This is the material for NTU-AI6129.

Here I will explain how to run app.py:

1, You should git clone my github:
```cmd
git clone https://github.com/CsO4/LLaVA_ChatBot
cd LLaVA_ChatBot
```

2, You should create a virtual environment:
```cmd
conda create -n llava python=3.10 -y
conda activate llava
```

3, Then you should follow the LLaVAChatBot class to pip install the necessary component:
```cmd
pip install git+https://github.com/haotian-liu/LLaVA.git@786aa6a19ea10edc6f574ad2e16276974e9aaa3a
```

4, At the last, you could try to run app.py:
```cmd
python app.py
```

Then it will load some parameters and checkpoints, and you could run the link in your local browser to get to the chatbot.
