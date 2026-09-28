"""Changed model: two models answering alike are still two samples.

The same question is put to a large model and a small one -- the shape a
distillation set is gathered in. On an easy turn they say the same thing.

The model is part of what identifies a node, so the second call starts a root
of its own rather than merging into the first. Nothing on the first path was
produced by the second model, and a dataset that means to train one of them, or
to distil one into the other, has to be able to tell them apart. Merging them
would report one generation where two models each produced one.

`scripts/model_distill.py` carries this through to the export, where it is one
row per model.
"""

from _capture import chat

question = [{"role": "user", "content": "Why does the auth test fail?"}]

teacher = chat(question, "The token expiry check is inverted.", model="teacher-70b")
student = chat(question, "The token expiry check is inverted.", model="student-7b")

print(f"teacher-70b: {teacher}")
print(f"student-7b:  {student}")
print(f"\nanswers identical: {teacher == student}. the graph should still show two roots.")
