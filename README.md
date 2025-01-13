# Realization_desicion_tree_cirrhosis

Признак:'Hepatomegaly' <= 0.0
  Признак:'Bilirubin' <= 1.0
    Признак:'Cholesterol' <= 253.0
      Признак:'Prothrombin' <= 10.3
        Класс: 2.0
      Признак: 'Prothrombin' > 10.3
        Класс: 3.0
    Признак: 'Cholesterol' > 253.0
      Признак:'Cholesterol' <= 280.0
        Класс: 2.0
      Признак: 'Cholesterol' > 280.0
        Класс: 3.0
  Признак: 'Bilirubin' > 1.0
    Признак:'Age' <= 19295.0
      Признак:'SGOT' <= 144.15
        Класс: 3.0
      Признак: 'SGOT' > 144.15
        Класс: 3.0
    Признак: 'Age' > 19295.0
      Признак:'Tryglicerides' <= 229.0
        Класс: 4.0
      Признак: 'Tryglicerides' > 229.0
        Класс: 3.0
Признак: 'Hepatomegaly' > 0.0
  Признак:'Albumin' <= 3.25
    Признак:'Platelets' <= 214.0
      Класс: 4.0
    Признак: 'Platelets' > 214.0
      Признак:'Bilirubin' <= 6.1
        Класс: 3.0
      Признак: 'Bilirubin' > 6.1
        Класс: 4.0
  Признак: 'Albumin' > 3.25
    Признак:'Prothrombin' <= 10.7
      Признак:'Copper' <= 73.0
        Класс: 3.0
      Признак: 'Copper' > 73.0
        Класс: 2.0
    Признак: 'Prothrombin' > 10.7
      Признак:'Age' <= 18352.0
        Класс: 4.0
      Признак: 'Age' > 18352.0
        Класс: 4.0

Accuracy: 0.5180722891566265
