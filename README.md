# mura-distillation

This project is about applying the knowledge distillation technique to the MURA dataset.

Steps to execute the code:

- Run teacher.py to train the DenseNet169 model.
- Then run student.py which will get the information from teacher model and train the MobileNet model.
- Run evaluation.py to see the performance of the model.
- In order to compare whether distillation works, run train_without_distillation.py model and evaluate with evaluation.py
