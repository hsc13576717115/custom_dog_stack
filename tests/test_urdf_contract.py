import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_DIR = PROJECT_ROOT / "ros2" / "src" / "custom_dog_description"
URDF_PATH = DESCRIPTION_DIR / "urdf" / "custom_dog.urdf"
LEGS = ("FR", "FL", "RR", "RL")
CALF_LOWER_LIMIT_RAD = math.radians(-162.0)
CALF_UPPER_LIMIT_RAD = math.radians(-48.0)


class UrdfContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = ET.parse(URDF_PATH).getroot()
        cls.links = {element.attrib["name"]: element for element in cls.robot.findall("link")}
        cls.joints = {element.attrib["name"]: element for element in cls.robot.findall("joint")}

    def test_expected_link_and_joint_sets(self):
        expected_links = {"base"}
        expected_joints = set()
        for leg in LEGS:
            expected_links.update(f"{leg}_{segment}" for segment in ("hip", "thigh", "calf", "foot"))
            expected_joints.update(f"{leg}_{segment}_joint" for segment in ("hip", "thigh", "calf", "foot"))
        self.assertEqual(set(self.links), expected_links)
        self.assertEqual(set(self.joints), expected_joints)

    def test_joint_types_and_tree(self):
        children = set()
        for name, joint in self.joints.items():
            expected_type = "fixed" if name.endswith("_foot_joint") else "revolute"
            self.assertEqual(joint.attrib["type"], expected_type)
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            self.assertIn(parent, self.links)
            self.assertIn(child, self.links)
            self.assertNotIn(child, children)
            children.add(child)
        self.assertEqual(set(self.links) - children, {"base"})

    def test_mesh_references_resolve(self):
        prefix = "package://custom_dog_description/"
        meshes = self.robot.findall(".//mesh")
        # Collision geometry uses engine-stable primitives; meshes are visual only.
        self.assertEqual(len(meshes), 17)
        for mesh in meshes:
            uri = mesh.attrib["filename"]
            self.assertTrue(uri.startswith(prefix), uri)
            self.assertTrue((DESCRIPTION_DIR / uri.removeprefix(prefix)).is_file(), uri)

    def test_every_canonical_link_has_one_primitive_collision(self):
        for name, link in self.links.items():
            collisions = link.findall("collision")
            self.assertEqual(len(collisions), 1, name)
            geometry = collisions[0].find("geometry")
            self.assertIsNotNone(geometry, name)
            self.assertEqual(len(geometry), 1, name)
            self.assertIn(geometry[0].tag, {"box", "cylinder", "sphere"}, name)

    def test_mass_and_inertia_are_positive(self):
        total_mass = 0.0
        for name, link in self.links.items():
            inertial = link.find("inertial")
            self.assertIsNotNone(inertial, name)
            mass = float(inertial.find("mass").attrib["value"])
            self.assertGreater(mass, 0.0, name)
            total_mass += mass

            values = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
            ixx, ixy, ixz = values["ixx"], values["ixy"], values["ixz"]
            iyy, iyz, izz = values["iyy"], values["iyz"], values["izz"]
            leading_minor_2 = ixx * iyy - ixy * ixy
            determinant = (
                ixx * (iyy * izz - iyz * iyz)
                - ixy * (ixy * izz - iyz * ixz)
                + ixz * (ixy * iyz - iyy * ixz)
            )
            self.assertGreater(ixx, 0.0, name)
            self.assertGreater(leading_minor_2, 0.0, name)
            self.assertGreater(determinant, 0.0, name)
        self.assertAlmostEqual(total_mass, 13.84916, places=5)

    def test_revolute_joint_limits_and_axes(self):
        revolute_count = 0
        for name, joint in self.joints.items():
            if joint.attrib["type"] != "revolute":
                continue
            revolute_count += 1
            limit = joint.find("limit").attrib
            self.assertLess(float(limit["lower"]), float(limit["upper"]), name)
            self.assertGreater(float(limit["effort"]), 0.0, name)
            self.assertGreater(float(limit["velocity"]), 0.0, name)
            axis = [float(value) for value in joint.find("axis").attrib["xyz"].split()]
            self.assertTrue(math.isclose(math.sqrt(sum(value * value for value in axis)), 1.0, abs_tol=1e-6), name)
        self.assertEqual(revolute_count, 12)

    def test_calf_limits_match_the_measured_folded_pose(self):
        for leg in LEGS:
            limit = self.joints[f"{leg}_calf_joint"].find("limit").attrib
            self.assertTrue(
                math.isclose(float(limit["lower"]), CALF_LOWER_LIMIT_RAD, abs_tol=1e-6),
                leg,
            )
            self.assertTrue(
                math.isclose(float(limit["upper"]), CALF_UPPER_LIMIT_RAD, abs_tol=1e-6),
                leg,
            )


if __name__ == "__main__":
    unittest.main()
