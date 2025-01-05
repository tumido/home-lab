#!/bin/sh

HA_PATH="${PWD}/ha"

echo "Updating custom components"

echo -n "- dreame-vacuum "
rm -rf ${HA_PATH}/custom_components/dreame_vacuum
mkdir -p ${HA_PATH}/custom_components/dreame_vacuum
curl -sSL "https://github.com/Tasshack/dreame-vacuum/releases/download/v2.0.0b16/dreame_vacuum.zip" | bsdtar -xf- -C ${HA_PATH}/custom_components/dreame_vacuum
echo "✅"
