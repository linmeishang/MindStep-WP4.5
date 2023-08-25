## runFarmDynfromBatch ----
#' Execute FarmDyn
#'
#' @description
#' `runFarmDynfromBatch()` does as it says in the function.
#'
#' @param FarmDynDir Directory where FarmDyn is located
#' @param IniFile Name of the IniFile
#' @param XMLFile Name of the XML file
#' @param BATCHDir Directory where the .batch file is located
#' @param BATCHFile Name of the .batch file
#'
#' @return Executes FarmDyn from R
#' @examples
#' TODO write example
#'
#' @seealso
#' *Globiom?
#'
#' @export runFarmDynfromBatch

runFarmDynfromBatch <- function(FarmDynDir, IniFile, XMLFile, BATCHDir, BATCHFile) {
#
  # make sub directories
  GUIDir <- paste(FarmDynDir,"GUI",sep="/")
  BATCHFilePath <- paste(BATCHDir, BATCHFile, sep = "\\")

  # General JAVA command
  javacmdstrg <- r"(java -Xmx1G -Xverify:none -XX:+UseParallelGC -XX:PermSize=20M -XX:MaxNewSize=32M -XX:NewSize=32M -Djava.library.path=jars -classpath jars\gig.jar de.capri.ggig.BatchExecution)"

  # append specific files to JAVA command
  javacmdparac <- paste(javacmdstrg,IniFile,XMLFile,BATCHFilePath,sep = " ")

  # create bat file
  runbat   = paste0(GUIDir,"/runfarmdyn.bat")
  if (file.exists(runbat)) x=file.remove(runbat)

  b = substr(runbat,1,2)
  b = c(b,paste('cd',gsub("/", "\\\\",GUIDir)))
  b = c(b,c("SET PATH=%PATH%;./jars"))
  b = c(b,javacmdparac)
  writeLines(b,runbat)
  rm(b)

  # execute farmdyn in batch mode
  system(runbat)


}
