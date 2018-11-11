# Based off: https://github.com/antonpaquin/Tensorflow-Lambda/blob/master/aws-build-lambda.sh

BUCKET_NAME="quickdraw-battle"
MODEL_KEY="quickdraw.model.h5"
LAMBDA_SOURCE_KEY="lambda-source.zip"
LAMBDA_PACKAGE_KEY="lambda.zip"
TEMP_FOLDER_NAME="predict_lambda"
BUILD_FOLDER_NAME="build"
TEST_FUNCTION_NAME="predict-test.py"

mkdir $TEMP_FOLDER_NAME
cd $TEMP_FOLDER_NAME

# Get lambda source (predict function, predict test file/data)
aws s3api get-object --bucket $BUCKET_NAME --key $LAMBDA_SOURCE_KEY $LAMBDA_SOURCE_KEY

# Get Keras h5 model
aws s3api get-object --bucket $BUCKET_NAME --key $MODEL_KEY $MODEL_KEY

# Install python 3.6
sudo yum install -y python36
# Make pip 3.6 available as "pip3"
sudo ln -s /usr/bin/pip-3.6 /usr/bin/pip3
# Start up the installation virtualenv
sudo env PATH=\$PATH pip3 install --upgrade virtualenv
virtualenv -p python3 env
source env/bin/activate

# Install dependencies (assumes pip for python 3)
pip3 install numpy keras tensorflow tqdm

# Install notify tools
sudo yum-config-manager --enable epel
sudo yum update -y
sudo yum install -y inotify-tools

mkdir build
mv $LAMBDA_SOURCE_KEY build
mv $MODEL_KEY build
pushd build
unzip $LAMBDA_SOURCE_KEY
rm $LAMBDA_SOURCE_KEY
popd

# We want to find out what files python actually uses in the process of running our script
# So we'll set up a listener for all accessed files in the virtualenv
inotifywait \
  -m \
  -e access \
  -o inotifywait.list \
  --format "%w%f" \
  -r \
  $VIRTUAL_ENV/lib/python3.6/site-packages/ &
  # Sleep to give inotify time to set up the watches
  sleep 1;

# Run predict test function
# inotify will then pick up which files are used during the prediction, and will be listed
# in the "inotifywait.list" file
pushd build
python3 $TEST_FUNCTION_NAME
kill $(pgrep inotifywait)
popd

# Copy over all of the used files to the build directory
pushd build
PATH_TRIM_AMOUNT=$(readlink -f $VIRTUAL_ENV/lib/python3.6/site-packages/ | awk '{print $1"/"}' | wc -m)
for f in $(cat ../inotifywait.list); do
  if [ -f $f ]; then
    # Trim each path to remove everything up to "site_packages"
    REL=$(dirname $f | cut -c $PATH_TRIM_AMOUNT-)
    mkdir -p $REL
    cp $f $REL
  fi
done

# Copy all the python files, because they're small and tend to break
# things if they're absent
pushd $VIRTUAL_ENV/lib/python3.6/site-packages/
find . -name "*.py" | cut -c 3- > $HOME/pydep.list
popd
for f in $(cat $HOME/pydep.list); do
  cp "$VIRTUAL_ENV/lib/python3.6/site-packages/$f" "../build/$f" 2>/dev/null
done
popd

# And start the final zipping process
pushd build
# Strip unnecessary symbols from binaries (shrinks about 90 MB)
find . -name "*.so" | xargs strip

# Remove crap
rm test-data easy_install.py six.py termcolor.py predict-test.py -fr
rm __pycache__ *.dist-info -fr

# Zip it up
zip -r9 lambda.zip *

# Push to S3 bucket
aws s3 cp lambda.zip s3://$BUCKET_NAME/$LAMBDA_PACKAGE_KEY
