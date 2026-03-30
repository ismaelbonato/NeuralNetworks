#!/bin/bash
# Script to generate .env file with current user's UID/GID

cat > .env << EOF
# Auto-generated environment variables for dev container
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER=$(whoami)
EOF

echo ".env file created with:"
cat .env
