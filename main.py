from fastapi import FastAPI, UploadFile
from utils.image import decode_image, get_score

app = FastAPI()


@app.get("/")
def read_root():
    return {"response": "Hello World"}


@app.post("/count/")
async def count(image: UploadFile):
    contents = await image.read()
    image = decode_image(contents)
    score = get_score(image)
    return {"score": score}
