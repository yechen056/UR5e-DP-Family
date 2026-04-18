from pymodbus.client.sync import ModbusTcpClient
from pymodbus.exceptions import ModbusException
import logging


class PgiTcpGripper:
    """
    TCP/IP controller for the PGI gripper via Modbus TCP.
    """
    def __init__(self):
        self.client = None
        self.unit = 1
        self.ip = None
        self.port = None

    def connect(self, ip, port, unit=1, timeout=1):
        self.ip = ip
        self.port = port
        self.unit = unit
        self.client = ModbusTcpClient(host=ip, port=port, timeout=timeout)
        connection = self.client.connect()
        return bool(connection)

    def disconnect(self):
        if self.client:
            self.client.close()

    def init_gripper(self):
        logging.basicConfig(level=logging.CRITICAL)
        try:
            self.client.write_register(0x0100, 1, unit=self.unit)
            while True:
                status = self.client.read_holding_registers(0x0200, 1, unit=self.unit)
                if status.registers[0] == 1:
                    break
        except ModbusException as e:
            print(e)

    def set_force(self, force):
        if 20 <= force <= 100:
            self.client.write_register(0x0101, force, unit=self.unit)
        else:
            print(f"力度值必须在20到100之间(单元ID: {self.unit})")

    def set_position(self, position):
        if 0 <= position <= 1000:
            result = self.client.write_register(0x0103, position, unit=self.unit)
            if result is None or getattr(result, "isError", lambda: False)():
                print(f"[PGI] Failed to set gripper position to {position} (unit={self.unit}).")
        else:
            print(f"位置值必须在0到1000之间(单元ID: {self.unit})")

    def set_speed(self, speed):
        if 1 <= speed <= 100:
            self.client.write_register(0x0104, speed, unit=self.unit)
        else:
            print(f"速度值必须在1到100之间(单元ID: {self.unit})")

    def read_current_position(self):
        result = self.client.read_holding_registers(0x0202, 1, unit=self.unit)
        if result is None or getattr(result, "isError", lambda: False)() or not hasattr(result, "registers"):
            print(f"[PGI] Failed to read current gripper position (unit={self.unit}).")
            return 0
        return result.registers[0]
