from setuptools import setup

package_name = 'hand_angle_node'

setup(
    name=package_name,  # ★ここが超重要。絶対に 'hand-angle-node' にしない！
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='staff@syblab.org',
    description='Hand angle node using MediaPipe to publish /hand_norm',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # hand_angle_node という実行ファイル名で main() を呼ぶ
            'hand_angle_node = hand_angle_node.hand_angle_node:main',
        ],
    },
)
