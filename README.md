# Image quality improvement project

This repository contains source code the source code for image quality
improvement pipeline. FastAPI app can be deployed for
appropriate web interface.

---
## Cloning the project
To clone the project with all external libraries and projects:
```shell
git clone --recurse-submodules <repo_link>
```

---

## Installation
Applicable for machine with CUDA12 installed.

To install the project run the following command:
```shell
make all
```

---

## Running FastAPI app
### Launching an app
To run FastAPI make sure to install project according to 
installation guidelines provided above and run the script:
```shell
cd src/api
python3 fastapi_app.py 
```
### Getting pipeline results
Go to "docs" section of api:

`<host>/docs`

To infererence pipeline on your image select it via post
`/upload_image` request.
You are going to get output image inline.


---

