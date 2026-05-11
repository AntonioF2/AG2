import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

ISLAND_MAP = {
    "Biscoe": 0,
    "Dream": 1,
    "Torgersen": 2,
}

SEX_MAP = {
    "FEMALE": 0,
    "MALE": 1,
}

SPECIES_MAP = {
    "Adelie": 0,
    "Chinstrap": 1,
    "Gentoo": 2,
}

INVERSE_SPECIES_MAP = {value: key for key, value in SPECIES_MAP.items()}

FEATURE_COLUMNS = [
    "island",
    "sex",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

TARGET_COLUMN = "species"


def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    # Reorder columns with features first and target last
    desired_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[desired_columns]

    df["island"] = df["island"].map(ISLAND_MAP)
    df["sex"] = df["sex"].map(SEX_MAP)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(SPECIES_MAP)

    return df


def create_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=42)
    if model_name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if model_name == "mlp":
        return MLPClassifier(max_iter=500, random_state=42)
    if model_name == "naive_bayes":
        return GaussianNB()
    raise ValueError(f"Modelo desconhecido: {model_name}")


def train_and_evaluate(df: pd.DataFrame, model_name: str):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = create_model(model_name)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, target_names=list(SPECIES_MAP.keys()))
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy, report


def parse_user_input():
    print("Digite os valores dos atributos do pinguim para predizer a espécie:")
    print("Ilhas disponíveis: Biscoe=0, Dream=1, Torgersen=2")
    print("Sexo: FEMALE=0, MALE=1")

    island = int(input("Ilha (0/1/2): ").strip())
    sex = int(input("Sexo (0/1): ").strip())
    culmen_length_mm = float(input("Comprimento do bico (mm): ").strip())
    culmen_depth_mm = float(input("Profundidade do bico (mm): ").strip())
    flipper_length_mm = float(input("Comprimento da nadadeira (mm): ").strip())
    body_mass_g = float(input("Massa corporal (g): ").strip())

    return [island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g]


def predict_species(model, scaler, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return INVERSE_SPECIES_MAP[prediction]


def main():
    parser = argparse.ArgumentParser(description="Classificador de pinguins usando palmerpenguins.csv")
    parser.add_argument(
        "--model",
        choices=["decision_tree", "knn", "mlp", "naive_bayes"],
        default="decision_tree",
        help="Escolha do classificador",
    )
    parser.add_argument(
        "--csv",
        default="palmerpenguins.csv",
        help="Caminho para o arquivo CSV do dataset",
    )
    args = parser.parse_args()

    df = load_and_prepare_data(args.csv)
    model, scaler, accuracy, report = train_and_evaluate(df, args.model)

    print("\n== Dados carregados e preparados ==")
    print(f"Amostras após remoção de valores ausentes: {len(df)}")
    print(f"Modelo treinado: {args.model}")
    print(f"Acurácia de teste: {accuracy:.4f}\n")
    print("Relatório de classificação:\n")
    print(report)

    user_features = parse_user_input()
    predicted_species = predict_species(model, scaler, user_features)
    print(f"\nEspécie prevista: {predicted_species}")


if __name__ == "__main__":
    main()
