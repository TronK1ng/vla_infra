"""M5 Round 01 writer skeleton tests: injected SDK, no real connection."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from vla_infra.contracts.pose import CartesianPose
from vla_infra.devices.flexiv_rdk import (
    FlexivRdkNrtCommandSink,
    NrtLimits,
    base_pose_to_rdk_world,
)
from vla_infra.devices.gn01 import Gn01CommandSink, Gn01Config
from vla_infra.devices.m5_auth import M5AuthorizationIssuer
from vla_infra.devices.preflight import M5PreflightRunner, REQUIRED_ENABLE_PROBES, REQUIRED_SECTIONS
from vla_infra.devices.real_session import FlexivRdkSession
from vla_infra.errors import (
    DeviceError,
    ErrorCode,
    RealHardwareDisabledError,
    SingleWriterViolationError,
)


class FakeMode:
    NRT_CARTESIAN_MOTION_FORCE = "NRT_CARTESIAN_MOTION_FORCE"


class FakeRobot:
    def __init__(self, serial):
        self.serial = serial
        self.calls = []
        self.enabled = False
        self.mode_value = None
        self.force_axes = None
        self._operational = False
        self._fault = False
        self._recovery = False
        self._estop = True
        self.mode_delay_polls = 0

    def operational(self):
        self.calls.append(("operational",))
        return self._operational

    def operational_status(self):
        return "READY" if self._operational else "NOT_READY"

    def fault(self):
        return self._fault

    def recovery(self):
        return self._recovery

    def estop_released(self):
        return self._estop

    def mode(self):
        self.calls.append(("mode",))
        return self.mode_value

    def Enable(self):
        self.calls.append(("Enable",))
        self.enabled = True
        self._operational = True

    def SwitchMode(self, mode):
        self.calls.append(("SwitchMode", mode))
        self.mode_value = mode

    def SetForceControlAxis(self, axes):
        self.calls.append(("SetForceControlAxis", list(axes)))
        self.force_axes = list(axes)

    def SendCartesianMotionForce(self, pose, wrench, velocity, max_linear_vel, max_angular_vel, max_linear_acc, max_angular_acc):
        self.calls.append(
            (
                "SendCartesianMotionForce",
                list(pose),
                list(wrench),
                list(velocity),
                [max_linear_vel, max_angular_vel, max_linear_acc, max_angular_acc],
            )
        )

    def Stop(self):
        self.calls.append(("Stop",))

    def ClearFault(self):  # must never be called automatically
        self.calls.append(("ClearFault",))


class FakeSdk:
    Mode = FakeMode

    def __init__(self):
        self.robots = []
        self.grippers = []
        self.tools = []

    def Robot(self, serial):
        robot = FakeRobot(serial)
        self.robots.append(robot)
        return robot

    def Gripper(self, robot):
        gripper = FakeGripper()
        gripper.robot = robot
        self.grippers.append(gripper)
        return gripper

    def Tool(self, robot):
        if self.tools:
            return self.tools[0]
        tool = FakeTool()
        tool.robot = robot
        self.tools.append(tool)
        return tool


def _sdk():
    return FakeSdk()


def _limits():
    return NrtLimits(0.02, 0.10, 0.05, 0.10)


def _identity():
    return np.eye(4)


def _claimed_composite(session, **kwargs):
    token = session.claim_runtime()
    return session.create_composite(owner_token=token, **kwargs)


def _pose(z=0.3):
    return CartesianPose(np.array([0.5, 0.0, z]), np.array([0.0, 0.0, 0.0, 1.0]))


def _auth(issuer=None):
    issuer = issuer or M5AuthorizationIssuer(
        station_id="S1", config_hash="cfg", secret=b"test-secret-32-bytes-long!!!!"
    )
    probes = {name: (lambda: {"status": "PASS"}) for name in REQUIRED_SECTIONS}
    enable = {name: (lambda: {"status": "PASS"}) for name in REQUIRED_ENABLE_PROBES}
    report = M5PreflightRunner(
        station_id="S1",
        config_hash="cfg",
        readonly_probes=probes,
        enable_probes=enable,
        clock_ns=lambda: 1000,
    ).run(allow_enable=True, operator_ack=True)
    return issuer, issuer.issue(report, operator_ack=True, now_ns=1000)


def _arm(sdk, transform=None):
    issuer, auth = _auth()
    sink = FlexivRdkNrtCommandSink(
        "R1",
        limits=_limits(),
        base_to_world=transform if transform is not None else _identity(),
        sdk=sdk,
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    sink._test_auth = auth
    sink._tool_readiness.mark_switched("G1")
    sink._tool_readiness.verify(name="G1", tcp_hash="tcp-hash", payload_hash="payload-hash")
    return sink


def _mark_tool(sink) -> None:
    sink._tool_readiness.mark_switched("G1")
    sink._tool_readiness.verify(name="G1", tcp_hash="tcp-hash", payload_hash="payload-hash")


def test_connect_does_not_enable_or_switch_mode() -> None:
    sdk = _sdk()
    sink = _arm(sdk)
    assert sdk.robots == []
    sink.connect()
    robot = sdk.robots[0]
    assert robot.calls == []
    assert sink.status.connected is True
    assert sink.status.enabled is False
    assert sink.status.mode_ready is False


def test_explicit_enable_and_nrt_pose_conversion() -> None:
    sdk = _sdk()
    sink = _arm(sdk)
    sink.connect()
    robot = sdk.robots[0]
    sink.claim_writer("executor")
    sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=1, now_ns=5)
    assert ("Enable",) in robot.calls
    assert ("SwitchMode", "NRT_CARTESIAN_MOTION_FORCE") in robot.calls
    assert ("SetForceControlAxis", [False] * 6) in robot.calls
    assert not any(call[0] == "ClearFault" for call in robot.calls)

    sink.command_pose(_pose(), epoch=1, now_ns=10)
    call = robot.calls[-1]
    assert call[0] == "SendCartesianMotionForce"
    assert call[1] == [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
    assert call[4] == [0.02, 0.10, 0.05, 0.10]
    assert sink.status.writes == 1


def test_base_to_world_transform_and_quaternion_order() -> None:
    angle = np.pi / 2.0
    transform = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 1.0],
            [np.sin(angle), np.cos(angle), 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pos, quat = base_pose_to_rdk_world(_pose(), transform)
    assert np.allclose(pos, [1.0, 2.5, 0.3])
    # 90deg z base rotation becomes wxyz [cos, 0, 0, sin]
    assert np.allclose(quat, [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    sdk = _sdk()
    sink = _arm(sdk, transform)
    sink.connect(); sink.claim_writer("executor"); sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=1, now_ns=5)
    sink.command_pose(_pose(), epoch=1, now_ns=10)
    command = sdk.robots[0].calls[-1][1]
    assert np.allclose(command[:3], [1.0, 2.5, 0.3])
    assert np.allclose(command[3:], [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def test_writer_requires_enable_and_single_writer() -> None:
    sdk = _sdk()
    sink = _arm(sdk)
    sink.connect()
    sink.claim_writer("executor")
    with pytest.raises(DeviceError):
        sink.command_pose(_pose(), epoch=1, now_ns=10)
    sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=1, now_ns=5)

    errors = []

    def other_thread():
        try:
            sink.command_pose(_pose(), epoch=1, now_ns=10)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=other_thread)
    thread.start(); thread.join()
    assert errors and isinstance(errors[0], SingleWriterViolationError)


def test_writer_sdk_failure_and_stop_failure_are_observable() -> None:
    sdk = _sdk()
    sink = _arm(sdk)
    sink.connect(); sink.claim_writer("executor"); sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=1, now_ns=5)
    robot = sdk.robots[0]

    def fail(*args, **kwargs):
        raise RuntimeError("bus lost")

    robot.SendCartesianMotionForce = fail
    with pytest.raises(DeviceError):
        sink.command_pose(_pose(), epoch=1, now_ns=10)
    assert "command_failed" in sink.status.last_error
    assert sink._capability.active is False
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        sink.command_pose(_pose(), epoch=1, now_ns=20)

    def stop_fail():
        raise RuntimeError("stop blocked")

    robot.Stop = stop_fail
    assert sink.stop_motion() is False
    assert sink.status.stop_failures == 1


def test_writer_without_sdk_requires_real_gate() -> None:
    sink = FlexivRdkNrtCommandSink("R1", limits=_limits(), base_to_world=_identity())
    with pytest.raises(RealHardwareDisabledError):
        sink.connect()


def test_writer_rejects_default_limits_and_non_base_pose() -> None:
    with pytest.raises(DeviceError):
        NrtLimits(0.5, 0.1, 0.05, 0.1)  # above skeleton ceiling
    sdk = _sdk()
    sink = _arm(sdk)
    sink.connect(); sink.claim_writer("executor"); sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=1, now_ns=5)
    with pytest.raises(DeviceError):
        sink.command_pose(
            CartesianPose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), frame="tcp"),
            epoch=1,
            now_ns=10,
        )


class FakeParams:
    name = "G1"
    min_width = 0.0
    max_width = 0.1
    min_vel = 0.001
    max_vel = 0.1
    min_force = 1.0
    max_force = 20.0


class FakeGripper:
    def __init__(self):
        self.moves = []
        self.stops = 0
        self.enabled = False
        self.fail_move = False
        self.fail_stop = False
        self.disable_calls = 0
        self._params = FakeParams()

    def params(self):
        return self._params

    def Enable(self, name):
        self.enabled = True

    def Move(self, width, velocity, force):
        if self.fail_move:
            raise RuntimeError("jam")
        self.moves.append((width, velocity, force))

    def Disable(self):
        self.disable_calls += 1
        self.enabled = False

    def Stop(self):
        if self.fail_stop:
            raise RuntimeError("stop jam")
        self.stops += 1


class FakeTool:
    def __init__(self):
        self.switches = []
        self.fail = False

    def Switch(self, name):
        if self.fail:
            raise RuntimeError("tool switch failed")
        self.switches.append(name)


class FakeGnSdk:
    def __init__(self):
        self.grippers = []
        self.tools = []

    def Gripper(self, robot):
        g = FakeGripper()
        g.robot = robot
        self.grippers.append(g)
        return g

    def Tool(self, robot):
        t = FakeTool()
        self.tools.append(t)
        return t


def test_gn01_normalized_width_dedup_range_and_stop() -> None:
    config = Gn01Config(0.0, 0.1, 0.01, 2.0)
    sdk = FakeGnSdk()
    _, auth = _auth()
    sink = Gn01CommandSink("G1", config, sdk=sdk, robot=object(), authorization_verifier=lambda a: True)
    sink.connect()
    _mark_tool(sink)
    sink._capability.grant(auth, 1)
    assert sdk.grippers[0].moves == []
    sink.enable(auth, authority_epoch=1)
    sink.claim_writer("executor")
    sink.command_gripper(0.0, epoch=1, now_ns=1)
    sink.command_gripper(0.5, epoch=1, now_ns=2)
    assert sdk.grippers[0].moves[-1] == (0.05, 0.01, 2.0)
    sink.command_gripper(0.5, epoch=1, now_ns=3)
    assert len(sdk.grippers[0].moves) == 2
    assert sink.status.dedup_skips == 1
    with pytest.raises(DeviceError):
        sink.command_gripper(1.2, epoch=1, now_ns=4)
    assert sink.stop() is True
    assert sdk.grippers[0].stops == 1

    sdk.grippers[0].fail_move = True
    with pytest.raises(DeviceError):
        sink.command_gripper(0.9, epoch=1, now_ns=5)
    assert sink._capability.active is False
    sdk.grippers[0].fail_stop = True
    assert sink.stop() is False


def test_gn01_without_sdk_requires_real_gate() -> None:
    sink = Gn01CommandSink("G1", Gn01Config(0.0, 0.1, 0.01, 2.0), robot=object(), authorization_verifier=lambda a: True)
    with pytest.raises(RealHardwareDisabledError):
        sink.connect()


def test_composite_partial_write_and_stop_are_observable() -> None:
    sdk = _sdk()
    issuer, auth = _auth()
    session = FlexivRdkSession("R1", sdk=sdk)
    session.connect()
    composite = _claimed_composite(session,
        limits=_limits(),
        base_to_world=_identity(),
        gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    arm = composite._arm
    gripper = composite._gripper
    composite.connect()
    _mark_tool(arm)
    composite.claim_writer("executor")
    arm.enable_and_switch_nrt(auth, authority_epoch=1, now_ns=5)
    gripper.enable(auth, authority_epoch=1)
    composite.command_pose(_pose(), epoch=1, now_ns=1)
    sdk.grippers[0].fail_move = True
    with pytest.raises(DeviceError):
        composite.command_gripper(0.5, epoch=1, now_ns=2)
    assert composite.status.partial_writes == 1
    assert composite._arm._capability.active is False
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        composite.command_pose(_pose(), epoch=1, now_ns=3)
    assert composite.stop_motion() is True
    assert arm.status.enabled is False
    assert gripper.status.enabled is False


def test_enable_waits_for_operational_and_verifies_actual_mode() -> None:
    sdk = _sdk()
    sink = _arm(sdk)
    sink.connect(); sink.claim_writer("executor")
    robot = sdk.robots[0]
    polls = {"n": 0}

    def delayed_operational():
        polls["n"] += 1
        return polls["n"] >= 3

    robot.operational = delayed_operational
    sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=2, timeout_s=0.2, sleep=lambda _: None)
    assert polls["n"] >= 3
    calls = [c[0] for c in robot.calls]
    assert calls.index("Enable") < calls.index("SwitchMode")


def test_enable_timeout_and_fault_stop_and_fail_closed() -> None:
    sdk = _sdk(); sink = _arm(sdk); sink.connect(); sink.claim_writer("executor")
    robot = sdk.robots[0]; robot.operational = lambda: False
    with pytest.raises(DeviceError) as exc:
        sink.enable_and_switch_nrt(sink._test_auth, timeout_s=0.01, sleep=lambda _: None)
    assert "operational" in exc.value.message
    assert ("Stop",) in robot.calls
    assert sink.status.enabled is False
    assert not any(call[0] == "ClearFault" for call in robot.calls)

    sdk2 = _sdk(); sink2 = _arm(sdk2); sink2.connect(); sink2.claim_writer("executor")
    robot2 = sdk2.robots[0]; robot2.fault = lambda: True
    with pytest.raises(DeviceError):
        sink2.enable_and_switch_nrt(sink2._test_auth, timeout_s=0.1, sleep=lambda _: None)
    assert ("Stop",) in robot2.calls
    assert sink2.status.enabled is False


def test_enable_mode_verification_failure_stops() -> None:
    sdk = _sdk(); sink = _arm(sdk); sink.connect(); sink.claim_writer("executor")
    robot = sdk.robots[0]; robot.mode = lambda: "WRONG"
    with pytest.raises(DeviceError) as exc:
        sink.enable_and_switch_nrt(sink._test_auth, timeout_s=0.01, sleep=lambda _: None)
    assert "NRT_CARTESIAN" in exc.value.message
    assert ("Stop",) in robot.calls


def test_enable_requires_verifier_authorization() -> None:
    sdk = _sdk()
    sink = FlexivRdkNrtCommandSink("R1", limits=_limits(), base_to_world=_identity(), sdk=sdk)
    sink.connect(); sink.claim_writer("executor")
    with pytest.raises(RealHardwareDisabledError):
        sink.enable_and_switch_nrt(object())


def test_command_epoch_and_inter_command_gap_are_bound() -> None:
    sdk = _sdk(); sink = _arm(sdk); sink.connect(); sink.claim_writer("executor")
    sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=3, now_ns=5)
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        sink.command_pose(_pose(), epoch=4, now_ns=10)
    sink.command_pose(_pose(), epoch=3, now_ns=10)
    with pytest.raises(DeviceError):
        sink.command_pose(_pose(), epoch=3, now_ns=10)
    with pytest.raises(DeviceError):
        sink.command_pose(_pose(), epoch=3, now_ns=10 + sink._max_inter_command_gap_ns + 1)
    sink.revoke()
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        sink.command_pose(_pose(), epoch=3, now_ns=1_000_000)


def test_gn01_tool_switch_is_explicit_and_params_subset_enforced() -> None:
    sdk = FakeGnSdk()
    _, auth = _auth()
    sink = Gn01CommandSink("G1", Gn01Config(0.0, 0.1, 0.01, 2.0), sdk=sdk, robot=object(), authorization_verifier=lambda a: True)
    sink.connect()
    sink._capability.grant(auth, 1)
    assert sdk.tools == []
    sink.enable(auth, authority_epoch=1)
    assert sdk.tools == []
    sink.switch_tool()
    assert sdk.tools[0].switches == ["G1"]

    sdk2 = FakeGnSdk()
    _, auth2 = _auth()
    bad = Gn01CommandSink(
        "G1", Gn01Config(0.0, 0.2, 0.01, 2.0), sdk=sdk2, robot=object(), authorization_verifier=lambda a: True
    )
    bad.connect(); bad._capability.grant(auth2, 1)
    with pytest.raises(DeviceError) as exc:
        bad.enable(auth2, authority_epoch=1)
    assert exc.value.details["disable_ok"] is True
    assert sdk2.grippers[0].disable_calls == 1
    assert bad.status.enabled is False


def test_gn01_requires_shared_authorization_and_epoch() -> None:
    sdk = FakeGnSdk()
    sink = Gn01CommandSink(
        "G1", Gn01Config(0.0, 0.1, 0.01, 2.0), sdk=sdk, robot=object(),
        authorization_verifier=lambda a: False,
    )
    _, auth = _auth()
    sink.connect()
    with pytest.raises(RealHardwareDisabledError):
        sink.enable(auth, authority_epoch=1)

    accepted = Gn01CommandSink(
        "G1", Gn01Config(0.0, 0.1, 0.01, 2.0), sdk=FakeGnSdk(), robot=object(),
        authorization_verifier=lambda a: True,
    )
    accepted.connect(); accepted._capability.grant(auth, 2); accepted.claim_writer("executor"); accepted.enable(auth, authority_epoch=2)
    _mark_tool(accepted)
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        accepted.command_gripper(0.5, epoch=1, now_ns=2)
    accepted.command_gripper(0.5, epoch=2, now_ns=3)
    accepted.revoke()
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        accepted.command_gripper(0.5, epoch=2, now_ns=4)


def test_shared_session_factory_uses_one_robot_handle() -> None:
    sdk = _sdk()
    issuer, _ = _auth()
    session = FlexivRdkSession("R1", sdk=sdk)
    assert session.connect() is None
    robot = sdk.robots[0]
    composite = _claimed_composite(session,
        limits=_limits(),
        base_to_world=_identity(),
        gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    assert composite._arm._robot is robot
    assert composite._gripper._robot is robot
    assert len(sdk.robots) == 1


def test_arm_and_gripper_share_authorization_denial() -> None:
    sdk = _sdk()
    session = FlexivRdkSession("R1", sdk=sdk)
    session.connect()
    composite = _claimed_composite(session,
        limits=_limits(),
        base_to_world=_identity(),
        gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: False,
    )
    composite.connect(); composite.claim_writer("executor")
    with pytest.raises(RealHardwareDisabledError):
        composite._arm.enable_and_switch_nrt(object(), timeout_s=0.01, sleep=lambda _: None)
    with pytest.raises(RealHardwareDisabledError):
        composite._gripper.enable(object(), authority_epoch=1)


def test_arm_revocation_blocks_shared_gripper() -> None:
    sdk = _sdk()
    issuer, auth = _auth()
    session = FlexivRdkSession("R1", sdk=sdk)
    session.connect()
    composite = _claimed_composite(session,
        limits=_limits(),
        base_to_world=_identity(),
        gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    composite.connect(); _mark_tool(composite._arm); composite.claim_writer("executor")
    composite._arm.enable_and_switch_nrt(auth, authority_epoch=1, now_ns=5)
    composite._gripper.enable(auth, authority_epoch=1)
    composite._arm.revoke()
    with pytest.raises(RealHardwareDisabledError):
        composite._gripper.command_gripper(0.5, epoch=1, now_ns=2)


def test_arm_stop_failure_still_revokes_capability() -> None:
    sdk = _sdk(); sink = _arm(sdk); sink.connect(); sink.claim_writer("executor")
    sink.enable_and_switch_nrt(sink._test_auth, authority_epoch=1, now_ns=5)
    robot = sdk.robots[0]
    robot.Stop = lambda: (_ for _ in ()).throw(RuntimeError("stop blocked"))
    assert sink.stop_motion() is False
    assert sink.status.enabled is False
    assert sink.status.mode_ready is False
    assert sink.status.physical_stop_confirmed is False
    assert sink._capability.active is False
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        sink.command_pose(_pose(), epoch=1, now_ns=10)


def test_gn01_stop_failure_still_revokes_and_move_rejected() -> None:
    _, auth = _auth()
    sdk = FakeGnSdk()
    sink = Gn01CommandSink(
        "G1", Gn01Config(0.0, 0.1, 0.01, 2.0), sdk=sdk, robot=object(),
        authorization_verifier=lambda a: True,
    )
    sink.connect(); sink._capability.grant(auth, 1); sink.claim_writer("executor"); sink.enable(auth, authority_epoch=1)
    _mark_tool(sink)
    sdk.grippers[0].fail_stop = True
    assert sink.stop() is False
    assert sink.status.enabled is False
    assert sink.status.physical_stop_confirmed is False
    assert sink._capability.active is False
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        sink.command_gripper(0.5, epoch=1, now_ns=2)


def test_tool_switch_rejects_old_authorization_after_regrant() -> None:
    sdk = _sdk(); issuer, auth = _auth(); session = FlexivRdkSession("R1", sdk=sdk); session.connect()
    composite = _claimed_composite(session,
        limits=_limits(), base_to_world=_identity(), gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    composite.connect(); composite.claim_writer("executor")
    composite._arm.enable_and_switch_nrt(auth, authority_epoch=1, now_ns=5)
    composite._gripper.enable(auth, authority_epoch=1)
    # Re-grant the shared capability with a different valid authorization/epoch.
    probes = {name: (lambda: {"status": "PASS"}) for name in REQUIRED_SECTIONS}
    enable_probes = {name: (lambda: {"status": "PASS"}) for name in REQUIRED_ENABLE_PROBES}
    report = M5PreflightRunner(
        station_id="S1", config_hash="cfg", readonly_probes=probes,
        enable_probes=enable_probes, clock_ns=lambda: 1000,
    ).run(allow_enable=True, operator_ack=True)
    other = issuer.issue(report, operator_ack=True, now_ns=1000)
    composite._arm._capability.grant(other, 2)
    with pytest.raises(RealHardwareDisabledError):
        composite._gripper.switch_tool()


def test_session_close_stops_and_revokes_or_fails_loudly() -> None:
    sdk = _sdk(); issuer, auth = _auth(); session = FlexivRdkSession("R1", sdk=sdk); session.connect()
    composite = _claimed_composite(session,
        limits=_limits(), base_to_world=_identity(), gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    composite.connect(); _mark_tool(composite._arm); composite.claim_writer("executor")
    composite._arm.enable_and_switch_nrt(auth, authority_epoch=1, now_ns=5)
    composite._gripper.enable(auth, authority_epoch=1)
    session.close()
    assert session.connected is False
    assert composite._arm.status.enabled is False
    assert composite._gripper.status.enabled is False
    assert any(call == ("Stop",) for call in sdk.robots[0].calls)

    sdk2 = _sdk(); issuer2, auth2 = _auth(); session2 = FlexivRdkSession("R1", sdk=sdk2); session2.connect()
    composite2 = _claimed_composite(session2,
        limits=_limits(), base_to_world=_identity(), gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer2.verify(a, now_ns=1000),
    )
    composite2.connect(); _mark_tool(composite2._arm); composite2.claim_writer("executor")
    composite2._arm.enable_and_switch_nrt(auth2, authority_epoch=1, now_ns=5)
    composite2._gripper.enable(auth2, authority_epoch=1)
    sdk2.robots[0].Stop = lambda: (_ for _ in ()).throw(RuntimeError("stop blocked"))
    with pytest.raises(DeviceError):
        session2.close()
    assert session2.connected is True


def test_gn01_cannot_self_grant_without_arm_lifecycle() -> None:
    _, auth = _auth()
    sdk = FakeGnSdk()
    sink = Gn01CommandSink(
        "G1", Gn01Config(0.0, 0.1, 0.01, 2.0), sdk=sdk, robot=object(),
        authorization_verifier=lambda a: True,
    )
    sink.connect(); _mark_tool(sink)
    with pytest.raises(RealHardwareDisabledError):
        sink.enable(auth, authority_epoch=1)
    assert sink._capability.active is False
    assert sdk.grippers[0].enabled is False


def test_tool_switch_failure_revokes_and_requires_verification() -> None:
    sdk = _sdk(); issuer, auth = _auth(); session = FlexivRdkSession("R1", sdk=sdk); session.connect()
    composite = _claimed_composite(session,
        limits=_limits(), base_to_world=_identity(), gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    composite.connect(); composite.claim_writer("executor")
    composite._arm.enable_and_switch_nrt(auth, authority_epoch=1, now_ns=5)
    composite._gripper.enable(auth, authority_epoch=1)
    # Successful switch is not a calibration verification.
    composite._gripper.switch_tool()
    with pytest.raises(DeviceError):
        composite.command_pose(_pose(), epoch=1, now_ns=10)
    composite._gripper.verify_tool(name="G1", tcp_hash="tcp", payload_hash="payload")
    composite.command_pose(_pose(), epoch=1, now_ns=20)

    # Switch failure revokes both arm and gripper authority.
    sdk.tools[0].fail = True
    with pytest.raises(DeviceError):
        composite._gripper.switch_tool()
    assert composite._arm._capability.active is False
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        composite.command_pose(_pose(), epoch=1, now_ns=30)
    with pytest.raises((DeviceError, RealHardwareDisabledError)):
        composite.command_gripper(0.5, epoch=1, now_ns=31)


def test_session_does_not_expose_raw_robot() -> None:
    sdk = _sdk(); session = FlexivRdkSession("R1", sdk=sdk); session.connect()
    assert not hasattr(session, "robot")
    assert not hasattr(session, "_test_robot_handle")
    assert session.connect() is None
    assert session.connected is True
    assert len(sdk.robots) == 1
    reader = session.state
    assert reader.started is False
    reader.start()
    assert reader.started is True


def test_session_allows_only_one_runtime_owner() -> None:
    sdk = _sdk()
    session = FlexivRdkSession("R1", sdk=sdk)
    token = session.claim_runtime()
    with pytest.raises(DeviceError):
        session.claim_runtime()
    with pytest.raises(DeviceError):
        session.release_runtime(object())
    session.release_runtime(token)
    session.claim_runtime()


def test_concurrent_session_claim_has_exactly_one_winner() -> None:
    sdk = _sdk(); session = FlexivRdkSession("R1", sdk=sdk)
    winners = []
    errors = []

    def claim():
        try:
            session.claim_runtime()
            winners.append(1)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1
    assert len(errors) == 7


def test_tool_identity_mismatch_blocks_command() -> None:
    from vla_infra.devices.tool_state import ToolReadiness

    sdk = _sdk(); issuer, auth = _auth()
    tool = ToolReadiness(expected_name="G1", expected_tcp_hash="a" * 64, expected_payload_hash="b" * 64)
    sink = FlexivRdkNrtCommandSink(
        "R1", limits=_limits(), base_to_world=_identity(), sdk=sdk,
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
        tool_readiness=tool,
    )
    sink.connect(); sink.claim_writer("executor")
    sink.enable_and_switch_nrt(auth, authority_epoch=1, now_ns=5)
    tool.mark_switched("G1")
    for kwargs in (
        {"name": "wrong", "tcp_hash": "a" * 64, "payload_hash": "b" * 64},
        {"name": "G1", "tcp_hash": "c" * 64, "payload_hash": "b" * 64},
        {"name": "G1", "tcp_hash": "a" * 64, "payload_hash": "c" * 64},
    ):
        with pytest.raises(DeviceError):
            tool.verify(**kwargs)
        with pytest.raises(DeviceError):
            sink.command_pose(_pose(), epoch=1, now_ns=10)
    assert not any(call[0] == "SendCartesianMotionForce" for call in sdk.robots[0].calls)
    tool.verify(name="G1", tcp_hash="a" * 64, payload_hash="b" * 64)
    sink.command_pose(_pose(), epoch=1, now_ns=20)


def test_abort_pair_does_not_skip_gripper_when_arm_blocks() -> None:
    sdk = _sdk(); issuer, auth = _auth(); session = FlexivRdkSession("R1", sdk=sdk); session.connect()
    composite = _claimed_composite(session,
        limits=_limits(), base_to_world=_identity(), gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    composite.connect(); composite.claim_writer("executor")
    release = threading.Event()
    sdk.robots[0].Stop = lambda: release.wait(2.0)
    with pytest.raises(DeviceError):
        session.abort_owned_resources(timeout_s=0.05)
    assert sdk.grippers[0].stops >= 1
    assert session.quarantined is True
    release.set()


def test_composite_partial_construction_cleans_arm() -> None:
    sdk = _sdk()
    session = FlexivRdkSession("R1", sdk=sdk)
    session.connect()

    def fail_gn(*args, **kwargs):
        raise RuntimeError("gripper construct failed")

    session._create_gn01_locked = fail_gn
    token = session.claim_runtime()
    with pytest.raises(RuntimeError):
        session.create_composite(
            limits=_limits(), base_to_world=_identity(), gn01_name="G1",
            gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
            authorization_verifier=lambda a: True,
            owner_token=token,
        )
    assert session._owned == []
    assert any(call == ("Stop",) for call in sdk.robots[0].calls)


def test_partial_construction_blocking_stop_quarantines_and_keeps_pending() -> None:
    class BlockingArm:
        def __init__(self):
            self.released = threading.Event()

        def revoke(self):
            return None

        def stop_motion(self, timeout_s):
            self.released.wait(2.0)
            return True

        def close(self):
            return None

    sdk = _sdk(); session = FlexivRdkSession("R1", sdk=sdk); session.connect()
    arm = BlockingArm()
    session._create_arm_locked = lambda **kwargs: arm
    session._create_gn01_locked = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("gn failed"))
    token = session.claim_runtime()
    with pytest.raises(DeviceError) as exc:
        session.create_composite(
            limits=_limits(), base_to_world=_identity(), gn01_name="G1",
            gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
            authorization_verifier=lambda a: True, cleanup_timeout_s=0.05,
            owner_token=token,
        )
    assert exc.value.code == ErrorCode.STOP_FAILED
    assert session.quarantined is True
    assert session.safety_gate.quarantined is True
    assert len(session._pending_owned) == 1
    with pytest.raises(DeviceError):
        arm._safety_gate.require_usable() if hasattr(arm, "_safety_gate") else session.safety_gate.require_usable()
    arm.released.set()


def test_session_quarantine_gate_blocks_existing_sink() -> None:
    sdk = _sdk(); session = FlexivRdkSession("R1", sdk=sdk); session.connect()
    token = session.claim_runtime()
    composite = session.create_composite(
        limits=_limits(), base_to_world=_identity(), gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: True, owner_token=token,
    )
    sink = composite._arm
    sink.connect(); sink.claim_writer("executor")
    session.quarantine("test_stop_timeout")
    with pytest.raises(DeviceError):
        sink.command_pose(_pose(), epoch=1, now_ns=10)
    assert not any(call[0] == "SendCartesianMotionForce" for call in sdk.robots[0].calls)


def test_unclaimed_session_cannot_create_any_writer() -> None:
    sdk = _sdk()
    session = FlexivRdkSession("R1", sdk=sdk)
    session.connect()
    with pytest.raises(DeviceError):
        session.create_arm(
            limits=_limits(),
            base_to_world=_identity(),
            authorization_verifier=lambda _: True,
        )
    with pytest.raises(DeviceError):
        session.create_gn01(
            name="G1",
            config=Gn01Config(0.0, 0.1, 0.01, 2.0),
            authorization_verifier=lambda _: True,
        )
    with pytest.raises(DeviceError):
        session.create_composite(
            limits=_limits(),
            base_to_world=_identity(),
            gn01_name="G1",
            gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
            authorization_verifier=lambda _: True,
        )
    assert session._owned == []
    assert session._pending_owned == []
    assert sdk.grippers == []


def test_quarantine_blocks_enable_switch_verify_and_commands() -> None:
    sdk = _sdk()
    issuer, auth = _auth()
    session = FlexivRdkSession("R1", sdk=sdk)
    session.connect()
    composite = _claimed_composite(
        session,
        limits=_limits(),
        base_to_world=_identity(),
        gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
    )
    composite.connect()
    composite.claim_writer("executor")
    session.quarantine("test_hard_gate")

    with pytest.raises(DeviceError):
        composite._arm.enable_and_switch_nrt(auth, authority_epoch=1, now_ns=5)
    with pytest.raises(DeviceError):
        composite._gripper.enable(auth, authority_epoch=1)
    with pytest.raises(DeviceError):
        composite._gripper.switch_tool()
    with pytest.raises(DeviceError):
        composite._gripper.verify_tool(
            name="G1", tcp_hash="a" * 64, payload_hash="b" * 64
        )
    with pytest.raises(DeviceError):
        composite.command_pose(_pose(), epoch=1, now_ns=10)
    with pytest.raises(DeviceError):
        composite.command_gripper(0.5, epoch=1, now_ns=10)

    robot_writes = {"Enable", "SwitchMode", "SetForceControlAxis", "SendCartesianMotionForce"}
    assert not any(call[0] in robot_writes for call in sdk.robots[0].calls)
    assert sdk.grippers[0].enabled is False
    assert sdk.grippers[0].moves == []
    assert sdk.tools == []
