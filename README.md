# Tests

Run the CPU smoke suite from the project root after installing `requirements.txt`:

```powershell
python -m unittest discover -s tests -v
```

The suite creates a tiny temporary ten-class fixture, performs only one GAN update and one epoch per tiny classifier, then checks checkpoint loading and evaluation outputs. It does not contact the network or use the full dataset. Its synthetic images are test fixtures, not project results.

