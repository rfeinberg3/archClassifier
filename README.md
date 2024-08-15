# ARM Endianness Classifier

Uses:

- Finding what architecture a piece of hardware has on it based only on the binary extracted. Assuming no headers are present in the binary.
- Without headers reverse-eng tools don't know what to do.

Ideas:

- Collect a variety of binary data from different ARM images, and train a classifier to distinguise characteristics about the architecture soley from binary data. e.g. distinguise a chunk of binary is the printf command.

## Design

To start, we'll a much more trivial problem. Classifying the endianness of an architecture based on the binary we're given.

1. Get Little Endian and Big Endian ARM setup with as many library downloads as possible.
2. Collect hex code from binaries of each architectures usr/bin directory.
3. Format the hex code + endian label as a supervised dataset.
4. Use Dev set to test. We want to make sure the model is training on similar amounts of little and big endian samples alternatley.
5. Train a model on the combined dataset of little and big endian hex code.
6. Validate on test set.

## Development

### Setting up Buildroot

Buildroot will be how we build different architectures with the data we need.

- We run buildroot in a docker container to avoid issues with host architecture and OS (in my case ARM and MacOS). See `Dockerfile` for details on dependencies.

### Building Architectures

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

- Execute the make commands to build ARM specified architecture

```bash
docker exec endl sh -c "make clean && make --keep-going"
```

- Run hexdump script and copy folder to local machine

```bash
docker exec endl sh -c "./hexdump.sh"
docker cp endl:/hexdump /hexdumps/hexdump_{archtype}
```

## Notes

### Reading the ELF (Executable and Linkable Format) header to see specs

This will tell you if a file is a binary executable.

```bash
readelf -h [file_name]
```

### Octal Dump (od)

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

### Buildroot menuconfig

Defining your own architecture with buildroot. This is how I obtained the configs for the desired architecture setup.

```bash
make clean
make menuconfig # Select architecture and packages of your choice in GUI.
make --keep-going
```
