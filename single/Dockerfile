# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python" ,"./Data.py"]
CMD ["python" ,"./XGBModel.py"]

##CMD python3 request.py --port 8050
##  docker build --tag mlproject-model:latest .
##  docker run -dit -p 127.0.0.1:8000:8080 IMAGEID
##  docker run -it -P IMAGEID
##  docker exec -it

# syntax=docker/dockerfile:1
#FROM python:3.8-slim-buster
#WORKDIR /app
#COPY requirements.txt requirements.txt
#RUN pip3 install -r requirements.txt
#COPY . .
#CMD ["python" ,"./Data.py"]
#CMD ["python" ,"./XGBModel.py"]
#RUN apt update
#CMD python3 request.py --port 8050
#  docker build --tag mlproject-model:latest .
#  docker run -dit -p 127.0.0.1:8000:8080 IMAGEID
#  docker exec -it‏