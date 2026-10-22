from afinn import Afinn
from fastapi import FastAPI

app = FastAPI(title="Course evaluaition sentiment API")

afinn_en = Afinn(language="en")
afinn_da = Afinn(language="da")



