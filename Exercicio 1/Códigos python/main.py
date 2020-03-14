import FuncoesAuxiliares as fa
def main():
    datasetPath = '../Kaggle/adult.csv'
    pathToSaveTable = '../Gráficos, tabelas e relatório/'
    tableName = 'describedAdult.xlsx'
    datasetDataframe = fa.getDataframeInSpecificFormat(datasetPath) 
    
    fa.getPercentageOfMissingValuesByColumns(datasetPath)
    removeColumnsWithMissingValues
if __name__ == "__main__":
    main()