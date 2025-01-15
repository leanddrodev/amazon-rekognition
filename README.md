# Amazon Rekognition - Detectando Celebridades em Imagens (Celebrity Recognition)

Este projeto faz parte do **Bootcamp Nexa - Análise Avançada de Imagens e Texto com IA na AWS**. O objetivo é reconhecer dentre várias faces as celebridades correspondentes. Para isso, será usado o SDK AWS para Python (Boto3) para trabalhar com o Amazon Rekognition.

O Amazon Rekognition é um serviço de análise visual baseado em deep learning para pesquisar, verificar e organizar milhões de imagens e vídeos.


# Imagens Selecionadas

Imagens antes do Amazon Rekognition fazer a análise de detecção de celebridades.

## Imagem 1
![bbc](https://github.com/user-attachments/assets/671c39ee-1684-431a-8b09-6d1f3a143872)

## Imagem 2
![msn](https://github.com/user-attachments/assets/09b05ebb-4aef-44a1-956e-48c06922346d)

## Imagem 3
![neymar-torcedores](https://github.com/user-attachments/assets/12e5b804-949a-408f-91d2-6793207de9bf)


# Celebridades Detectadas

Resultado depois do Amazon Rekognition processar as imagens e detectar as celebridades.

```python
from pathlib import Path

import boto3
from mypy_boto3_rekognition.type_defs import (
    CelebrityTypeDef,
    RecognizeCelebritiesResponseTypeDef,
)
from PIL import Image, ImageDraw, ImageFont

client = boto3.client("rekognition")


def get_path(file_name: str) -> str:
    return str(Path(__file__).parent / "images" / file_name)


def recognize_celebrities(photo: str) -> RecognizeCelebritiesResponseTypeDef:
    with open(photo, "rb") as image:
        return client.recognize_celebrities(Image={"Bytes": image.read()})


def draw_boxes(image_path: str, output_path: str, face_details: list[CelebrityTypeDef]):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Ubuntu-R.ttf", 20)

    width, height = image.size

    for face in face_details:
        box = face["Face"]["BoundingBox"]  # type: ignore
        left = int(box["Left"] * width)  # type: ignore
        top = int(box["Top"] * height)  # type: ignore
        right = int((box["Left"] + box["Width"]) * width)  # type: ignore
        bottom = int((box["Top"] + box["Height"]) * height)  # type: ignore

        confidence = face.get("MatchConfidence", 0)
        if confidence > 90:
            draw.rectangle([left, top, right, bottom], outline="red", width=3)

            text = face.get("Name", "")
            position = (left, top - 20)
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(bbox, fill="red")
            draw.text(position, text, font=font, fill="white")

    image.save(output_path)
    print(f"Imagem salva com resultados em : {output_path}")


if __name__ == "__main__":
    photo_paths = [
        get_path("bbc.jpg"),
        get_path("msn.jpg"),
        get_path("neymar-torcedores.jpg"),
    ]

    for photo_path in photo_paths:
        response = recognize_celebrities(photo_path)
        faces = response["CelebrityFaces"]
        if not faces:
            print(f"Não foram encontrados famosos para a imagem: {photo_path}")
            continue
        output_path = get_path(f"{Path(photo_path).stem}-resultado.jpg")
        draw_boxes(photo_path, output_path, faces)
```

## Imagem 1
![bbc-resultado](https://github.com/user-attachments/assets/e5a09944-78fa-46b2-a52e-6baa3defa939)

## Imagem 2
![msn-resultado](https://github.com/user-attachments/assets/c4c64e15-3313-4c2e-a2e9-436c85be0b08)

## Imagem 3
![neymar-torcedores-resultado](https://github.com/user-attachments/assets/6d3e01c0-bcd5-4921-ac06-531b44d0f077)
