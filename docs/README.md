# Documentation

## Model

[ryfye181/distilbert_endian_classifier](https://huggingface.co/ryfye181/distilbert_endian_classifier)

- distilled BERT finetuned on endianness dataset below.
- 77.1% accuracy on test set.
- See `classifiers/notebooks/distilbert_endian_classifier_training.ipynb` for details on training.

## Dataset

[ryfye181/endianness](https://huggingface.co/datasets/ryfye181/endianness)

Created by extracting all the binaries from the target directory of different architectures and converting them into hex format.

Dataset Specs:

- download size: 1.62 GB
- train: 7.05M feature rows
- test: 1.05M feature rows
- feature columns: 2 (hexdump of binary + architecture endianness label)
- labels: little endian, big endian

### Linux Commands for Binary Data Collection

#### Reading the ELF (Executable and Linkable Format) header to see specs

This will tell you if a file is a binary executable.

```bash
readelf -h [file_name]
```

#### Octal Dump (od)

Used to get the hex format of each binary file.

- Flags I used:

```bash
od -j64 -w64 -v -x -An [file_name]
```

- j64: Jump 64 bytes before reading to avoid reading header info
- v: Don't delete duplicate information
- x: select hexidecimal as output (each 4 number unit is 2 bytes)
- An: Don't output offset column
- (optional) w64: write 64 columns (for 32 bytes per column)
- [Read More](https://www.geeksforgeeks.org/od-command-linux-example/)

#### Buildroot menuconfig

Defining your own architecture with buildroot. This is how I obtained the configs for the desired architecture setup.

```bash
make clean
make menuconfig # Select architecture and packages of your choice in GUI.
make --keep-going
```
