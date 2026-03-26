from dataclasses import dataclass


@dataclass
class FleetConfig:
    """
    Cấu hình fleet cho RPP-mTD.

    Attributes:
        num_trucks (K):     số trucks
        drones_per_truck (M): số drones trên mỗi truck
        max_flight_time (τ): thời gian bay tối đa của drone (giờ)
        delta (δ):          số hop tối đa truck đi trong khi drone bay
        truck_speed:        tốc độ truck (km/h)
        drone_speed:        tốc độ drone (km/h)
        depot_id:           node id của depot

    Vehicle ID convention (1-based):
        Truck k  -> id = (k-1)*(M+1) + 1         (k = 1..K)
        Drone m của truck k -> id = (k-1)*(M+1) + 1 + m  (m = 1..M)

    Ví dụ K=2, M=2:
        T1=1, D1_1=2, D1_2=3
        T2=4, D2_1=5, D2_2=6
    """
    num_trucks: int
    drones_per_truck: int
    max_flight_time: float
    delta: int
    truck_speed: float = 40.0   # km/h
    drone_speed: float = 80.0   # km/h
    depot_id: int = 0

    def __post_init__(self):
        assert self.num_trucks >= 1
        assert self.drones_per_truck >= 0
        assert self.max_flight_time > 0
        assert self.delta >= 1
        assert self.truck_speed > 0
        assert self.drone_speed > 0

    @property
    def total_vehicles(self) -> int:
        """Tổng số vehicle (trucks + drones)."""
        return self.num_trucks * (1 + self.drones_per_truck)

    def truck_id(self, k: int) -> int:
        """ID của truck k (k bắt đầu từ 1)."""
        assert 1 <= k <= self.num_trucks
        return (k - 1) * (self.drones_per_truck + 1) + 1

    def drone_id(self, k: int, m: int) -> int:
        """ID của drone m trên truck k (k, m bắt đầu từ 1)."""
        assert 1 <= k <= self.num_trucks
        assert 1 <= m <= self.drones_per_truck
        return (k - 1) * (self.drones_per_truck + 1) + 1 + m

    def all_truck_ids(self) -> list[int]:
        return [self.truck_id(k) for k in range(1, self.num_trucks + 1)]

    def all_drone_ids(self) -> list[int]:
        return [
            self.drone_id(k, m)
            for k in range(1, self.num_trucks + 1)
            for m in range(1, self.drones_per_truck + 1)
        ]

    def all_vehicle_ids(self) -> list[int]:
        return sorted(self.all_truck_ids() + self.all_drone_ids())

    def is_truck(self, vid: int) -> bool:
        return vid in self.all_truck_ids()

    def is_drone(self, vid: int) -> bool:
        return vid in self.all_drone_ids()

    def parent_truck_id(self, drone_vid: int) -> int:
        """Truck cha của drone có id drone_vid."""
        for k in range(1, self.num_trucks + 1):
            for m in range(1, self.drones_per_truck + 1):
                if self.drone_id(k, m) == drone_vid:
                    return self.truck_id(k)
        raise ValueError(f"Vehicle {drone_vid} không phải drone")

    def drone_ids_of_truck(self, truck_vid: int) -> list[int]:
        """Tất cả drone ids thuộc truck truck_vid."""
        for k in range(1, self.num_trucks + 1):
            if self.truck_id(k) == truck_vid:
                return [self.drone_id(k, m) for m in range(1, self.drones_per_truck + 1)]
        raise ValueError(f"Vehicle {truck_vid} không phải truck")

    def system_ids(self, k: int) -> list[int]:
        """Tất cả vehicle ids của truck system k (truck + drones của nó)."""
        return [self.truck_id(k)] + [self.drone_id(k, m) for m in range(1, self.drones_per_truck + 1)]
