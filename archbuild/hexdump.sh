#!/bin/bash

rm -rf "$(pwd)/hexdump"
mkdir "$(pwd)/hexdump"

# Recursively writes the contents of each binary file in a the target architectures
# dirtectory to a respective text file in hexidecimal format.
function hex_dump {
    local path="$1"

    if [ -d "$path" ]; then
        for item in "$path"/*; do
            hex_dump "$item"
        done
    elif [ -f "$path" ]; then
        local output_file="$(pwd)/hexdump/$(basename "$path").txt"
        od -j64 -w64 -v -x -An "$path" > "$output_file"
    fi
}

# Call the function with desired directory
hex_dump "$(pwd)/output/target"
