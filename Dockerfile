# Base Image
FROM tensorflow/tensorflow:latest

COPY . /petq-modelserver

# Set working directory
WORKDIR /petq-modelserver
RUN mkdir ~/.pip && printf "[global]\nindex-url=http://ftp.daumkakao.com/pypi/simple\ntrusted-host=ftp.daumkakao.com" > ~/.pip/pip.conf
RUN /usr/local/bin/python -m pip install --upgrade pip && pip install --upgrade -r ../requirments.txt --use-feature=2020-resolver

CMD [ "python", "app.py" ]