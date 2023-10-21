# Computer-Version-final-project
install liborsa

1.Conda Install
conda install -c conda-forge librosa

2.If you’re using a Python 3.5 environment in conda, you may run into trouble with the numba dependency. This can be avoided by installing from the numba conda channel before installing librosa:
conda install -c numba numba

3.jupyter notebook(install in code line)
pip install liborsa

# Audio
each length is limit to 10 second .mp3 format

filename is groupname+number ex: road1

four group 1.ambulance 2.police 3.road 4.crush

# Image
ex:road
![TD_road1](https://github.com/BelindaNTHU/Computer-Version-final-project/assets/146788377/aa35f0ec-4c22-47ff-bddd-33c98844cc3b)
![FD_road1](https://github.com/BelindaNTHU/Computer-Version-final-project/assets/146788377/79f030ed-59fb-400a-9197-46e5a37ac44f)
ex:ambulance
![timeD_ambulance1](https://github.com/BelindaNTHU/Computer-Version-final-project/assets/146788377/ce237bb4-32e6-4388-b9a4-b4adbbc9ce7b)
![frequencyD_ambulance1](https://github.com/BelindaNTHU/Computer-Version-final-project/assets/146788377/b177fc61-0cfa-41eb-88ed-796356bf1d22)

# REFERENCE
feature_extraction.py

https://github.com/cs-hasibul/Music-Feature-Extraction-in-Python/blob/main/Audio_Data_Analysis_Using_Deep_Learning_with_Python.ipynb
