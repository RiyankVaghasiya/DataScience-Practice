from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lte=10)

new_student = {'name': "Riyank Vaghasiya",'age':20, 'email': "riyank@gmail.com",'cgpa': 10}

student = Student(**new_student)

print(student)