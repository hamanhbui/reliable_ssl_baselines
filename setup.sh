#Download the datasets
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14FaoU1AND3TpX2oJdMzidAO59zajTn7O' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14FaoU1AND3TpX2oJdMzidAO59zajTn7O" -O datasets.zip && rm -rf /tmp/cookies.txt
unzip datasets.zip
rm -rf datasets.zip