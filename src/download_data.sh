if [ -d $1/data ] && echo "Directory $1/data already exists."; then
  exit 1
else mkdir $1/data
fi
mkdir $1/data/esnli
wget https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_dev.csv -P $1/data/esnli
wget https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_train_1.csv -P $1/data/esnli
wget https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_train_2.csv -P $1/data/esnli
mkdir $1/data/creak
wget https://raw.githubusercontent.com/yasumasaonoe/creak/main/data/creak/train.json -P $1/data/creak
wget https://raw.githubusercontent.com/yasumasaonoe/creak/main/data/creak/dev.json -P $1/data/creak
mkdir $1/data/comve
wget https://raw.githubusercontent.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation/master/ALL%20data/train.csv -P $1/data/comve
wget https://raw.githubusercontent.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation/master/ALL%20data/dev.csv -P $1/data/comve
wget https://storage.googleapis.com/feb-data/data.zip -P $1/data
unzip $1/data/data.zip 
rm -rf $1/data/ECQA-Dataset 
rm -rf $1/data/SenseMaking
mv $1/data/SBIC $1/data/sbic
rm $1/data/data.zip $1/data/sbic/SBIC.v2.tst.modified.csv
mv $1/data/sbic/SBIC.v2.dev.modified.csv $1/data/sbic/dev.csv
mv $1/data/sbic/SBIC.v2.trn.modified.csv $1/data/sbic/train.csv