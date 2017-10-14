#!/bin/bash
#
# usage: deploy [machine-name] [container-name]
#

set -euo pipefail

#
# Functions
#

# Convenience function for logging text to stdout in pretty colors.
function log {
    local message="$@"

    local green="\\033[1;32m"
    local reset="\\033[0m"

    if [[ $TERM == "dumb" ]]
    then
        echo "${message}"
    else
        echo -e "${green}${message}${reset}"
    fi
}

# Check if a container with a given name already exists
function container_exists {
    local container_name=$1

    if [[ -z "$(docker ps -q -f name=${container_name})" ]]
    then false
    else true
    fi
}

# Check if a container with a given name exists and has status 'exited'
function container_is_exited {
    local container_name=$1

    if [[ -z "$(docker ps -aq -f status=exited -f name=${container_name})" ]]
    then false
    else true
    fi
}

# Get the image tag version based on the git SHA
function version {
    git rev-parse HEAD | cut -c1-8
}

# Build the Docker image and deploy it to a given Docker machine and
# give the container a specific name.
# Remove any containers that already exist with the given name.
function deploy {
    local machine_name=$1
    local container_name=$2

    log "Connection to machine"
    eval $(docker-machine env ${machine_name})

    log "Building image"
    docker build --tag komma:$(version) .

    if container_exists ${container_name}
    then
        log "Deleting exiting container"
        docker rm -f ${container_name}
    fi

    log "Starting containter"
    docker run \
        --detach \
        --publish 80:80 \
        --name ${container_name} \
        komma:$(version)

    log "Now running on http://$(docker-machine ip ${machine_name})"
}

#
# Main
#

machine_name=${1:-komma}
container_name=${2:-komma_flask}

deploy ${machine_name} ${container_name}
