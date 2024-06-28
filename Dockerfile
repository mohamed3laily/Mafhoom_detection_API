FROM python:3.8.10

WORKDIR /code

# Install system dependencies required for PyTorch and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt
RUN camel_data -i morphology-db-all

COPY ./ /code/

EXPOSE 4400

CMD ["gunicorn", "--bind", "0.0.0.0:4400" , "app:app"]
