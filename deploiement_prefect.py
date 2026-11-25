"""Étape 5 : déploiements Prefect (lancement planifié 1 fois par jour)."""

from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, install_flow

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours à 02:00
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            tags=["training", "mlops"],
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            tags=["evaluation", "mlops"],
        ),
        install_flow.to_deployment(
            name="ml-pipeline-install",
            tags=["setup", "mlops"],
        ),
    )
