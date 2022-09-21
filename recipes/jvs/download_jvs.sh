S1="19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt";
S2="jvs_ver1.zip";

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$S1" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$S1" -O $S2;
unzip jvs_ver1.zip

