#!/usr//bin/env python
import replicate

model = replicate.models.get("jimothyjohn/colmap")


def GetColmap():
    archive = model.predict(video="https://whatagan.s3.amazonaws.com/LionStatue.MOV")
    assert archive.startswith("https://")
    return True


def RunTests():
    assert GetColmap()
    return True


if RunTests():
    print("Tests passed!")
else:
    print("Tests failed...")
