 FROM tensorflow/tensorflow
 WORKDIR /app
 COPY . .
 #RUN pip install keras
 CMD ["python3", "UNet++_MSOF_model.py"]