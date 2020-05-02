oc_build_image:
	oc process --filename=openshift/oc-image-build-template.yaml \
		| oc apply -f -

oc_trigger_build:
	oc start-build "anomaly-detector-image"

oc_delete_image:
	oc process --filename=openshift/oc-image-build-template.yaml \
		--param APPLICATION_NAME="anomaly-detector-image" \
		| oc delete -f -

oc_deploy_app:
	oc process --filename=openshift/oc-deployment-template.yaml \
		--param PROMETEUS_URL="http://prometheus-k8s-monitoring.192.168.99.117.nip.io/" \
		| oc apply -f -

oc_delete_app:
	oc process --filename=openshift/oc-deployment-template.yaml \
	--param PROMETEUS_URL="http://prometheus-k8s-monitoring.192.168.99.117.nip.io/" \
		| oc delete -f -

run_app_pipenv:
	pipenv run python app.py

run_test_pipenv:
	pipenv run python test_model.py

run_app:
	python app.py

run_test:
	python test_model.py
