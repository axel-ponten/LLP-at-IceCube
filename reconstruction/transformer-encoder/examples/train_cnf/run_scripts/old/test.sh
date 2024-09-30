list1=(9 99 999)
list2=("/data/abc/" "/data/def/" "/data/ghi/")
list3=("apple" "banana" "cherry")

len=${#list1[@]}

for ((i=0;i<len;i++))
do
    echo ${list1[i]}
    echo ${list2[i]}
    echo ${list3[i]}
done