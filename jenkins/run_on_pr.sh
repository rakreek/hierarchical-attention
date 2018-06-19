#!/bin/bash
# Following http://conflu.allstate.com/display/COE/2017/07/24/Continuous+Integration+in+Deep+Learning%3A+Part+II%3A+Jenkins

source ~/.config/oauth/github_status_api_oauth.sh
export GITSTATUS_URL="https://github.allstate.com/api/v3/repos/D3NLP/fntk/statuses/$GIT_COMMIT?access_token=$OAUTH"
export JENKINS_URL="http://lxe0636.allstate.com:8080/job/fntk/$BUILD_NUMBER/console"

# even though these can be pulled from the keys of the hash maps below, it is necessary to
# maintain a separate list in order to loop over the steps in the correct order
steps=("BUILD" "TEST" "DOC")

# commands is hash map of commands to run at each build/test step
declare -A commands=(
    ["BUILD"]="pip install -e ${WORKSPACE} --user"
    ["TEST"]="make test-cover"
    ["DOC"]="make doc"
)

# commands is hash map of human readable descriptions to post to github
declare -A descs=(
    ["BUILD"]="build"
    ["TEST"]="unit and integration tests, with coverage report"
    ["DOC"]="rebuild documentation"
)

# commands is hash map of statuses (pending, success, failure) for each build/test step
declare -A statuses

# commands is hash map of exit codes for each build/test step
declare -A exitcodes

git_status_update() {
    for step in "${!descs[@]}"; do 
        echo "$step  ${descs[$step]}"
        curl $GITSTATUS_URL \
        -H "Content-Type: application/json" \
        -X POST \
        -d "{\"context\": \"${descs[$step]}\", \"state\": \"${statuses[$step]}\", \"description\": \"Jenkins\", \"target_url\": \"$JENKINS_URL\"}";
    done;
}

# prebuild step initializes all status to "pending"
for step in "${steps[@]}"; do
    statuses[$step]="pending"
    echo "$step  ${statuses[$step]}"
done
git_status_update

# build stage
for step in "${steps[@]}"; do
    echo "$step"
    if [ $step = "BUILD" ]; then 
        eval ${commands[$step]}
        cd ${WORKSPACE}/
    else
        eval ${commands[$step]}
    fi
    exitcodes[$step]=$?
    echo "$step  ${exitcodes[$step]}"
done

# postbuild step updates statuses based on exitcodes
for step in "${steps[@]}"; do
    if [ ${exitcodes[$step]} -eq 0 ];
    then
    statuses[$step]="success"
    else
    statuses[$step]="failure"
    fi
    echo "$step ${exitcodes[$step]} ${statuses[$step]}"
done
git_status_update

for step in "${steps[@]}"; do
    if [ ${exitcodes[$step]} -eq 0 ] then exit 1 fi
done
