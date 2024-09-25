# ARM Endianness Classifier

Uses:

- Finding what architecture a piece of hardware has on it based only on the binary extracted. Assuming no headers are present in the binary.
- Without headers reverse-eng tools don't know what to do.

Idea:

- Collect a variety of binary data from images of different architectures, and train a classifier to distinguise characteristics about the architecture soley from binary data. e.g. distinguise a chunk of binary is the printf command.

## Design

We'll start with a much more trivial problem. Classifying the endianness of an architecture based on the program binaries we can extract from its.

1. Get Little Endian and Big Endian ARM setup with as many library downloads as possible.
2. Collect binaries from each architectures usr/bin directory in hex format.
3. Format the hex code + endian label as a supervised dataset.
4. Use dev set for hyperparameter tuning and early stop. We also want to  ensure the model is training on similar amounts of little and big endian samples alternatley to avoid overfitting or underfitting on a certain class.
5. Fine-tune a classification model on the combined dataset of little and big endian hex code (e.g. BERT).
6. Validate with test set to determine accuracy and loss deltas.

## Development of ArchType Dataset

### Setting up Buildroot

Buildroot will be how we build different architectures, which contain the binary data we need.

- Download buildroot by cloning the repo into the `archbuild` directory.

```bash
cd archbuild
git clone https://gitlab.com/buildroot.org/buildroot.git
```

### Building Architectures

We run buildroot in a docker container to avoid issues with host architecture and OS (in my case ARM and MacOS). See `archbuild/Dockerfile` for details on dependencies.

#### Container Setup

- Use the Dockerfile to setup the container

```bash
cd archbuild
docker build -t br .
```

- Create the container with tty and detach. (And interactive for debugging if needed later).

```bash
docker run -it -d --name endl br
```

#### Building ARM architecture

- Copy in desired setup config

```bash
docker cp configs/config_{arctype} endl:/buildroot/.config
```

- Execute the make commands to build ARM specified architecture. This took several hours on my machine (Chip: Apple M2, Memory: 24 GB).

```bash
docker exec endl sh -c "make clean && make --keep-going"
```

- Run `hexdump.sh` script and copy folder to local machine

```bash
docker exec endl sh -c "./hexdump.sh"
docker cp endl:/hexdump /hexdumps/hexdump_{archtype}
```

## Building Dataset

See `datasetbuild/src/data2set.py` for details on dataset creation.

## Dataset Analysis

See `datasetbuild/src/set_analyzer.py` for details on dataset analysis.

## Classification

See `classifiers/notebooks/` for details on classification model development.

## TODO

- [] Gradio app for endianness classification
- [] Add all libraries in config to each arch build
- [] Add more architectures
- [] Develop tokenization method for binary data
- [] Try out other models
