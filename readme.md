# AI Scene Classifier
This project uses a Flask web interface and an ONNX Runtime model to classify images into 6 categories (buildings, forest, glacier, mountain, sea, street).

## Dockerization
The project is fully containerized. A `Dockerfile` is included to ensure the application environment is consistent across all machines.

### Instructions for the Evaluator:
To launch the application using Docker, run the following commands in the project root directory:

1. Build the image:
   `docker build -t ai-scene-app .`
2. Run the container:
   `docker run -p 5000:5000 ai-scene-app`
3. Access the application at: `http://localhost:5000`

*Note: If local disk-I/O locks occur during the build process, the application can also be launched locally using `python app.py`.*
# if you run  flask api then you run these commands 
py -m pip install -r requirements.txt    
python app.py or python test.py 
http://127.0.0.1:5000/  # <--- link 



# if you run   docker image then you run these commands 
# open the docker desktop : 
docker build -t ai-scene-app .
docker run -p 5000:5000 ml-project
http://localhost:5000

docker desktop image repostary :  https://hub.docker.com/repository/docker/ahmedraza0011/urooj/general
docker run -p 5000:5000 ai-scene-app
and then link generate 
http://localhost:5000
