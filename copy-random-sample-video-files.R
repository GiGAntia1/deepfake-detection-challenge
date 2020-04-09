
# Copy random video files from a class-specific
# training data directory to the corresponding
# validation data directory
x = list.files("/mnt/extHDD/Kaggle/videos/training/REAL",full.names=T)
x_n = length(x)
x_idx = sample(x_n,400)

for(i in x_idx){
  file.copy(x[i],"/mnt/extHDD/Kaggle/videos/validation/REAL")
  file.remove(x[i])
}




# Identify files in relevant directories
# Identify video file names in 0-2
setwd("/mnt/ubuntuHDD/Kaggle")
dir_names = paste0("dfdc_train_part_",c(0,1,2,6,12,18,24,30,34))
files = vector()
for(i in dir_names){
  
  files = c(files,paste0("/mnt/ubuntuHDD/Kaggle/",i,"/",list.files(i)))
  
}

# Identify and separate metadata JSON files
meta = files[grepl("json",files)]
files = files[!grepl("json",files)]

# Read metadata
library(data.table)
library(jsonlite)
meta_dt = data.table(FILE=character(),LABEL=character(),SPLIT=character(),ORIGINAL=character())
for(i in meta){
  
  x = fromJSON(i)
  dt = rbindlist(x, fill=T)
  dt[,FILE:=names(x)]
  setnames(dt,c("LABEL","SPLIT","ORIGINAL","FILE"))
  setcolorder(dt,"FILE")
  meta_dt = rbindlist(list(meta_dt,dt))
  
}

# Copy files to the respective validation folders
# Remove video files from training directory
n = nrow(meta_dt)
for(i in 1:n){
  
  file_name = meta_dt[i,FILE]
  file_label = meta_dt[i,LABEL]
  file_source_path = paste0("/mnt/extHDD/Kaggle/videos/training/",file_label,"/",file_name)
  file_dest_dir = paste0("/mnt/extHDD/Kaggle/videos/validation/",file_label,"/")
  
  if(file.exists(file_source_path)){
    
    file.copy(file_source_path, file_dest_dir)
    file.remove(file_source_path)
    
  }else{
    
    print(paste("File",file_source_path,"does not exist."))
    
  }
  
  # Print the completion progress percentage
  if(i %% 100 == 0){
    perc = paste0(sprintf("%1.2f",100*i/n),"% Complete...")
    print(perc)
  }
  
}
