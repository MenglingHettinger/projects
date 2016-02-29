#' Random Forest Regressor using the randomForest Package with ordinal encoding and missing value imputation
#'
#' @Author: Mark Steadman \email{mark@datarobot.com}
#' @Copyright: 2014 DataRobot Inc
#' @rtype: Regression
#' @row_limit: 20000
#'
#' This program is free software; you can redistribute it and/or
#' modify it under the terms of the GNU General Public License
#' as published by the Free Software Foundation; either version 2
#' of the License, or (at your option) any later version.
#'
#' This program is distributed in the hope that it will be useful,
#' but WITHOUT ANY WARRANTY; without even the implied warranty for
#' MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#' GNU General Public License for more details.
#'
#' You should have received a copy of the GNU General Public License
#' along with this program; if not, write to the Free Software
#' Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

modelfit <- function(response, data, extras) {
  # Fits a model using randomForest
  #
  # Args:
  #   response: a vector containing the response values (regression problem)
  #   data: a data frame containing the training data
  #   extras: a list containing any extras to be passsed
  #
  # Returns:
  #   model: a list containing all state information needed to
  #          predict on  new data
  
  #  Here we initialize some variables to control the fit

  # Any categorical variables with cardinality higher than CARD_MAX will be encoded
  # as numeric rather than as a factor level
  CARD_MAX <- 12 

  # Control the number of trees used to build the forest
  if (nrow(data) < 10000) {
    NTREES <- 350
  } else {
    NTREES <- 250
  }

  # Control the mininum leaf size
  if (nrow(data) < 15000) {
    NODESIZE <- 5
  } else { if(nrow(data) < 30000) {
      NODESIZE <- 10
    } else {
      NODESIZE <- 20
    }
  }
 
  # Any categorical variable that occurs fewer than SUPPORT_MIN times will be 
  # grouped together in a category by themselves
  SUPPORT_MIN <- max(5,nrow(data)/1000)
  
  # Set the random number generator seed
  set.seed(543543)
  
  # Record the time used for preprocessing for now
  start_time <- proc.time()
  
  # Load the gdata package which provides the maplevels functionality
  # to encode factor levels
  library(gdata)

  # Load the randomForest package which implements the Breiman and
  # Cutler randomForest
  library(randomForest)

  # remove NA targets
  data <- data[!is.na(response),]
  response <- response[!is.na(response)]

  
  # model will contain all information needed at predict time
  model <-  list()
  
  # Categorical variable preprocessing
  
  # Separate out character variables and encode as factor levels
  mchar_cols <- names(which(sapply(data, FUN=is.character)))
  data[mchar_cols] <- lapply(data[mchar_cols], FUN=as.factor)
  
  # Keep cols if not all the same and not all different
  col_levels <- lapply(data[mchar_cols], FUN=nlevels)
  char_cols_used <- names(which((col_levels > 1) & (col_levels < (0.95 * nrow(data)))))
  
  
  # Create empty vectors and lists in case we skip the main processing loop
  mchar_cols_low_card <-vector()
  mchar_cols_high_card <- vector()
  mfactor_maps <- list()
  
  fit_data <- list()
  
  # Check that at least one passes 
  if (length(char_cols_used) > 0) {
    
    # create a list of the column names for the high-cardinality categorical variables
    mchar_cols_high_card <- names(which(col_levels >= CARD_MAX))
    mchar_cols_high_card <- mchar_cols_high_card[which(mchar_cols_high_card %in% char_cols_used)]
    
    # And another list with any low-cardinality categorical variables
    if(length(mchar_cols_high_card) > 0) {
      mchar_cols_low_card <- setdiff(char_cols_used, mchar_cols_high_card)
    } else {
      mchar_cols_low_card <- char_cols_used 
    }

    # Reorder based on frequency for high_cardinality variables
    # Also combine the levels which have low support
    if(length(mchar_cols_high_card) > 0) {
      for(i in 1:length(mchar_cols_high_card)) {
        cur_col <- mchar_cols_high_card[i]
        freq_table <- table(data[mchar_cols][[cur_col]])
        low_support <- names(freq_table)[as.numeric(freq_table) < SUPPORT_MIN]
        table_ord <- order(as.numeric(freq_table), decreasing=TRUE)
        data[mchar_cols][[cur_col]] <-  factor(data[mchar_cols][[cur_col]], levels(data[mchar_cols][[cur_col]])[table_ord])
        levels(data[mchar_cols][[cur_col]])[levels(data[mchar_cols][[cur_col]]) %in% low_support] <- "--Low Support--"
      }
    }
    # We ensure that we have an addNA level to hold any new levels found
    data[char_cols_used] <- lapply(data[char_cols_used], FUN=addNA)
    
    # Record factor levels for later
    mfactor_maps <- lapply(data[mchar_cols], FUN=mapLevels, codes=FALSE)
    
    # Encode as an integer for high card
    data[mchar_cols_high_card] <- lapply(data[mchar_cols_high_card], FUN=as.integer)
  }
  
  # Numeric variable preprocessing

  
  # Record numeric columns that have variation
  mnum_cols <- colnames(data)[!(colnames(data) %in% mchar_cols)]
  
  # Ensure that the variables are defined
  mmedians <- list()
  mvcols <- c()
  
  # Check that we have at least one numeric variale
  if (length(mnum_cols) > 0) {
    
    # Missing Value Imputation for Numeric Variables
    mmedians <- sapply(data[mnum_cols], FUN=median, na.rm=TRUE)
    
    n_num_cols <- length(mnum_cols)
    
    # Recalculate median if only a single val
    # And replace NA with a different value
    single_val <- which(sapply(data[mnum_cols], FUN=function(X) { return((max(X, na.rm=TRUE) == min(X, na.rm=TRUE))) & is.finite(max(X, na.rm=TRUE))}))
    if(length(single_val) > 0) {
        for(i in 1:length(single_val)){
          val <- max(data[mnum_cols[single_val[i]]], na.rm=TRUE)
          if(is.finite(val) & any(is.na(data[mnum_cols[single_val[i]]]))) {
            mmedians[single_val[i]] <- val - 1
          }
        
        }
    }

    # Loop over the numeric columns
    for(i in 1:n_num_cols) {

      # Remove columns with only NA
      if (is.na(mmedians[i])) {
        data[mnum_cols[i]] <- NULL

      # Otherwise create a missing value indicator flag
      } else {
        indicator <- is.na(data[mnum_cols[i]])
        if (length(which(indicator)) > 0) {
          data[mnum_cols[i]][indicator] <- as.numeric(mmedians[i])
          colname <- paste(mnum_cols[i], ".mvi", sep="")
          mvcols <- append(mvcols, colname)
          data[colname] <- as.numeric(indicator)
        }
      }
  }
    
    # Update mnum_cols keep only the columns we keep
    mnum_cols <- colnames(data)[!(colnames(data) %in% union(mchar_cols, mvcols))]
    mmedians <- mmedians[mnum_cols]
    
  }
  
  mchar_cols <- union(mchar_cols_low_card, mchar_cols_high_card)
  
  # Create model matrix for fitting
  mtdata <- terms(~. - 1, data=data[c(mchar_cols_low_card, mchar_cols_high_card, mnum_cols, mvcols)])
  data2 <- model.matrix(mtdata, data[c(mchar_cols_low_card, mchar_cols_high_card, mnum_cols, mvcols)])
  
  print("Preprocessing time:")
  print(proc.time() - start_time)
  
  # Model fitting
  
  # Fit the model
  MTRY <- min(max(floor(ncol(data2)/3),1), 50)
  mrf <- randomForest(x=data2, y=response, ntree=NTREES, nodesize=NODESIZE, mtry=MTRY, replace=FALSE)
  
  # Store the model and the information needed for preprocessing at predict time
  model <- list(rf=mrf, medians=mmedians, factor_maps=mfactor_maps, char_cols=mchar_cols,
                num_cols=mnum_cols, char_cols_low_card=mchar_cols_low_card, char_cols_high_card=mchar_cols_high_card,
                mvcols=mvcols, tdata=mtdata)
  return(model)
};

modelpredict <- function(model, data) {
  # Function to make predictions using a fitted randomForest model
  #
  # Args:
  #   model : list
  #     Contains stored state information
  #   data : data.frame
  #     Contains data to make predictions on
  #
  # Returns:
  # predictions : vector 
  #   Contains predicted values    
  library(gdata)
  library(randomForest)
  
  # Categorical Variable Preprocessing

  # Encode factor levels to match the same order as at fit time
  data[model$char_cols] <- lapply(data[model$char_cols], FUN=as.character)
  
  predict_data <- list()
  if (length(model$char_cols) > 0) {
    for(i in 1:length(model$char_cols)) {
      cur_col <-model$char_cols[i]
      mapLevels(data[[cur_col]]) <- model$factor_maps[[cur_col]]

      # Any new levels will be encoded with the same level as any NA in the training data
      is.na(data[[cur_col]][is.na(as.numeric(data[[cur_col]]))]) <- TRUE
    }
    
    # And ordinal encode
    data[model$char_cols_high_card] <- lapply(data[model$char_cols_high_card], FUN=as.integer)
  };
  
  # Numeric variable preprocessing

  if (length(model$num_cols) > 0) {
    
    # Missing Value Imputation    
    for(i in 1:length(model$num_cols)) {
      indicator <- is.na(data[model$num_cols[[i]]])
      data[model$num_cols[[i]]][indicator] <- as.numeric(model$medians[[i]])
      
      # Create missing value indicator columns, if they were used at fit time
      colname <- paste(model$num_cols[[i]], ".mvi", sep="")
      if(colname %in% model$mvcols) {
        data[colname] <- as.integer(indicator)
      }
    }
    
    # If no missing values the predict method still needs the indicator columns
    missing_cols <- model$mvcols[which(!(model$mvcols %in% colnames(data)))]

    if(length(missing_cols) > 0){
      data[missing_cols] <- rep.int(0, nrow(data))
    }
  }
  
  # Model prediction code
  
  # Create the data Matrix
  data2 <- model.matrix(model$tdata, data[c(model$char_cols_low_card, model$char_cols_high_card, model$num_cols, model$mvcols)])
  print(colnames(data2)[which(! (c(model$char_cols_low_card, model$char_cols_high_card, model$num_cols, model$mvcols)) %in% colnames(data2))])
  # Return predictions
  predictions <- predict(model[['rf']], newdata=data2)
  
};

datarobot.run("R Random Forest Regressor Reference Model", modelfit, modelpredict)
