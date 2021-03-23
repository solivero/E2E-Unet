FROM tensorflow/tensorflow
WORKDIR /app
RUN pip install tensorflow-io
RUN ["/bin/bash"]