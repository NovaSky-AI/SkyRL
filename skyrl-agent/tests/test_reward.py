import os
import ray 
 
os.environ["ALLHANDS_API_KEY"] = "sycao-sandbox-remote"
os.environ["SANDBOX_REMOTE_RUNTIME_API_URL"] = "http://146.235.215.207:8000"

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.reward_manager.swebench import SWEBenchRewardManager
import numpy as np

incorrect_instance_id = "getmoto__moto-7456"
incorrect_patch = "diff --git a/models.py b/models.py\nnew file mode 100644\nindex 000000000..3319def44\n--- /dev/null\n+++ b/models.py\n@@ -0,0 +1,23 @@\n+from moto.core.models import BaseBackend\n+import boto3\n+\n+class ResilienceHubBackend(BaseBackend):\n+    def __init__(self, region_name):\n+        self.region_name = region_name\n+        self.policies = {}\n+\n+    def create_resiliency_policy(self, policy_name, policy_description, policy, tier):\n+        # Generate a unique ARN for the policy\n+        policy_arn = f\"arn:aws:resiliencehub:{self.region_name}:123456789012:resiliencypolicy/{policy_name}\"\n+        self.policies[policy_arn] = {\n+            \"policyName\": policy_name,\n+            \"policyDescription\": policy_description,\n+            \"policy\": policy,\n+            \"tier\": tier,\n+            \"policyArn\": policy_arn\n+        }\n+        return self.policies[policy_arn]\n+\n+\n+# Initialize the backend for us-east-1\n+resiliencehub_backends = {\"us-east-1\": ResilienceHubBackend(\"us-east-1\")}\n\\ No newline at end of file\ndiff --git a/moto/resiliencehub b/moto/resiliencehub\nnew file mode 100644\nindex 000000000..e69de29bb\ndiff --git a/reproduce_error.py b/reproduce_error.py\nnew file mode 100644\nindex 000000000..1ae93f615\n--- /dev/null\n+++ b/reproduce_error.py\n@@ -0,0 +1,33 @@\n+import boto3\n+from moto import mock_aws\n+import os\n+import pytest\n+\n+# Mocked AWS Credentials for moto\n+os.environ[\"AWS_ACCESS_KEY_ID\"] = \"testing\"\n+os.environ[\"AWS_SECRET_ACCESS_KEY\"] = \"testing\"\n+os.environ[\"AWS_SECURITY_TOKEN\"] = \"testing\"\n+os.environ[\"AWS_SESSION_TOKEN\"] = \"testing\"\n+os.environ[\"AWS_DEFAULT_REGION\"] = \"us-east-1\"\n+\n+def test_resiliencehub():\n+    with mock_aws():\n+        resiliencehub_client = boto3.client(\"resiliencehub\", region_name=\"us-east-2\")\n+        # Create a policy\n+    response = resiliencehub_client.create_resiliency_policy(\n+            policyName=\"mock-resilience-hub-basic-webapp\",\n+            policyDescription=\"Mocked resiliency policy\",\n+            policy={\n+                \"AZ\": {\"rpoInSecs\": 600, \"rtoInSecs\": 600},\n+                \"Hardware\": {\"rpoInSecs\": 600, \"rtoInSecs\": 600},\n+                \"Region\": {\"rpoInSecs\": 3600, \"rtoInSecs\": 14400},\n+                \"Software\": {\"rpoInSecs\": 600, \"rtoInSecs\": 600}\n+            },\n+            tier=\"Standard\",\n+        )\n+        print(\"create_policy response\")\n+        print(response)\n+        return response.get(\"policy\", {}).get(\"policyArn\")\n+\n+if __name__ == \"__main__\":\n+    test_resiliencehub()\n\\ No newline at end of file\ndiff --git a/moto/resiliencehub b/moto/resiliencehub\nnew file mode 100644\nindex 000000000..55fa1d462\n--- /dev/null\n+++ b/moto/resiliencehub\n@@ -0,0 +1,5 @@\n+__init__.py\n+exceptions.py\n+models.py\n+responses.py\n+urls.py\n\\ No newline at end of file\ndiff --git a/reproduce_error.py b/reproduce_error.py\nnew file mode 100644\nindex 000000000..87905c8e6\n--- /dev/null\n+++ b/reproduce_error.py\n@@ -0,0 +1,17 @@\n+import boto3\n+from moto import mock_aws\n+\n+@mock_aws\n+def test_resiliencehub():\n+    client = boto3.client(\"resiliencehub\", region_name=\"us-east-2\")\n+    response = client.create_resiliency_policy(\n+        policyName=\"mock-resilience-hub-basic-webapp\",\n+        policyDescription=\"Mocked resiliency policy\",\n+        policy={\"AZ\": {\"rpoInSecs\": 600, \"rtoInSecs\": 600}, \"Hardware\": {\"rpoInSecs\": 600, \"rtoInSecs\": 600}, \"Region\": {\"rpoInSecs\": 3600, \"rtoInSecs\": 14400}, \"Software\": {\"rpoInSecs\": 600, \"rtoInSecs\": 600}},\n+        tier=\"Standard\",\n+    )\n+    print(\"create_policy response\")\n+    print(response)\n+\n+if __name__ == \"__main__\":\n+    test_resiliencehub()\n\\ No newline at end of file"

correct_instance_id = "getmoto__moto-4986"
correct_patch = "diff --git a/moto/eks/models.py b/moto/eks/models.py\nindex 89fff7930..924916695 100644\n--- a/moto/eks/models.py\n+++ b/moto/eks/models.py\n@@ -140,7 +140,7 @@ class Cluster:\n         self.version = version or DEFAULT_KUBERNETES_VERSION\n\n         self.client_request_token = client_request_token\n-        self.encryption_config = encryption_config\n+        self.encryption_config = [encryption_config] if encryption_config else []\n         self.name = name\n         self.resources_vpc_config = resources_vpc_config\n         self.role_arn = role_arn\n@@ -162,7 +162,7 @@ class Cluster:\n         yield \"clientRequestToken\", self.client_request_token\n         yield \"platformVersion\", self.platformVersion\n         yield \"tags\", self.tags\n-        yield \"encryptionConfig\", self.encryption_config\n+        yield \"encryptionConfig\", self.encryption_config if isinstance(self.encryption_config, list) else []\n\n     def isActive(self):\n         return self.status == \"ACTIVE\"\ndiff --git a/reproduce_issue.py b/reproduce_issue.py\nnew file mode 100644\nindex 000000000..d6fa9a4dd\n--- /dev/null\n+++ b/reproduce_issue.py\n@@ -0,0 +1,37 @@\n+import boto3\n+import json\n+from moto import mock_iam, mock_eks\n+import os\n+\n+# Mock AWS credentials\n+os.environ['AWS_ACCESS_KEY_ID'] = 'testing'\n+os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'\n+os.environ['AWS_SECURITY_TOKEN'] = 'testing'\n+os.environ['AWS_SESSION_TOKEN'] = 'testing'\n+\n+@mock_iam\n+@mock_eks\n+def test_create_eks_cluster():\n+    # Create IAM role\n+    iam = boto3.client('iam', region_name='us-east-1')\n+    response = iam.create_role(\n+        RoleName='test-cluster',\n+        AssumeRolePolicyDocument='{\"Version\": \"2012-10-17\", \"Statement\": [{\"Action\": \"sts:AssumeRole\", \"Effect\": \"Allow\", \"Principal\": {\"Service\": \"eks.amazonaws.com\"}}]}'\n+    )\n+    role_arn = response['Role']['Arn']\n+    print(f'Created IAM Role ARN: {role_arn}')\n+\n+    # Create EKS cluster and get encryptionConfig\n+    eks = boto3.client('eks', region_name='us-east-1')\n+    response = eks.create_cluster(\n+        name='test',\n+        roleArn=role_arn,\n+        resourcesVpcConfig={\n+            'subnetIds': ['subnet-12345678']  # Assuming this subnet exists in the environment\n+        }\n+    )\n+    encryption_config = response['cluster']['encryptionConfig']\n+    print('Cluster encryptionConfig:', json.dumps(encryption_config))\n+\n+if __name__ == '__main__':\n+    test_create_eks_cluster()\n\\ No newline at end of file"

from verl.utils import hf_tokenizer
tokenizer = hf_tokenizer("Qwen/Qwen2.5-Coder-7B-Instruct")

n = 128
data = DataProto(
    batch=TensorDict({"responses": torch.randn(2*n)}, batch_size=2*n),
    non_tensor_batch={
        "data_source": np.array(["swe-gym", "swe-gym"]*n, dtype=object),
        "ability": np.array(["coding", "coding"]*n, dtype=object),
        "instance": np.array([{"instance_id": incorrect_instance_id}, {"instance_id": correct_instance_id}]*n, dtype=object),
        "git_patch": np.asarray([incorrect_patch, correct_patch]*n, dtype=object)
    }
)

@ray.remote(num_cpus=1)
def task(data):
    manager = SWEBenchRewardManager(None, None, None)
    score, metric = manager.verify_ray(data)
    # print(score)
    assert sum(score) == 1.0*n, "Not passed."
    print(f"Pass reward manager test.")

ray.get(task.remote(data))