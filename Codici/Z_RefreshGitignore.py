import os
"""faccio in modo di creare un .gitignore nella directory "primoprogetto" ed
escludere tutti i file di dimensione > 100 MB
"""



def fun_ignore_list(parentpath : str = "/home/silvia/Documents/GitHub/Progetto_Lenovo",
                    list_ignore : list = [], len_ancestor : int = 0, max_bytes: int = 95*1000*1024):
    """For example, if you want to ignore the Qualchefile.txt file located in the root directory,
    you would add the following line to .gitignore: /Qualchefile.txt, per sottocartelle add: /SottoCartella/Qualchefile.txt"""            
    # TODO ATTENTO AL '/' nel path, senza di esso ignora tutt i file nominati Qualchefile.txt


    # COMPLETO 
    if len_ancestor == 0:       # non mi serve "/home/silvia/Desktop.../A/B/C" ma solo "/A/B/C". Salvo lunghezza di "/home/silvia/D..."
        len_ancestor = len(parentpath)

    for it in os.scandir(parentpath):
        # print(it.path)
        if it.is_dir():
            # print(it.path)
            fun_ignore_list(parentpath=it.path, list_ignore=list_ignore, max_bytes=max_bytes, len_ancestor=len_ancestor)
        if not it.is_dir():
            if os.path.getsize(it.path) >= max_bytes:
                list_ignore.append(it.path[len_ancestor:])
                print(f'NON SONO UNA CARTELLA: {it.path} una con tutte stelle nella vita {os.path.getsize(it.path)}')
    return list_ignore



def create_gitignore(fold_gitignore: str, **kwargs):
        # Verifico se già esiste .gitignore, nel caso conservo la versione (.gitignore_ori)
        fold_gitignore_files = os.scandir(fold_gitignore)
        names = []
        for entry in fold_gitignore_files:
            names.append(entry.path)
        print ("\n\nnomi: ", names)
        if f"{fold_gitignore}/.gitignore" in names:
            os.rename(f"{fold_gitignore}/.gitignore", f"{fold_gitignore}/.gitignore_ori")
            print("\n\n\n############################## HO RINOMINATO ##############################")

        # creo .gitignore
        f = open(f"{fold_gitignore}/.gitignore","w")
        list_ignore = fun_ignore_list(parentpath=fold_gitignore)
        print("\n",list_ignore)
        for i in list_ignore:
            if i[1] != "." and i[1:5] != "venv":
                f.write(f"{i}\n")
        f.close()
 

create_gitignore(fold_gitignore="/home/silvia/Documents/GitHub/primoprogetto")



