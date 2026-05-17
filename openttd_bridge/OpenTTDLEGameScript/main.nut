class OpenTTDLEGameScript extends GSController {
    step = 0;
    coal_goal = null;
    routes = null;
    next_route_number = 1;

    function Start();
    function SendObservation(reason);
    function ReadAdminEvents();
    function ExecuteAction(message);
    function ChooseTownPair();
    function FindBuildableTileNear(center_tile);
    function BuildRoadBurstNear(center_tile, label);
    function BuildRoadBurst(start_tile, label);
    function TryRoad(tile_a, tile_b);
    function CollectCargoPairs(label, limit);
    function AddPairSorted(pairs, pair, limit);
    function GoalSummary();
    function RouteSummaries();
    function CargoWaitingSummaries();
    function StationRatingSummaries();
    function IndustryCargoInputs(limit);
    function IndustryCargoOutputs(limit);
    function ExecuteCoalRoute(action);
    function CreateVirtualRoute(source_id, destination_id, cargo_id, source_tile, destination_tile, delivery_rate, profit_rate);
    function ExecuteCargoRoute(action, action_type);
    function ExecuteAddVehicles(action);
    function ExecuteInspectBottlenecks();
    function ExecuteBorrowOrRepay(action);
    function ExecuteWait(action);
    function FindRoute(route_id);
    function FindCargoForPair(source_id, destination_id, preferred_cargo);
    function FindRoadStopNearIndustry(industry_id, target_tile);
    function ChooseFrontTile(tile, target_tile);
    function FindRoadPath(start_tile, end_tile);
    function PathTileUsable(tile, start_tile, end_tile);
    function TileKey(tile);
    function BuildRoadPath(path);
    function CountRoadConnections(path);
    function FindAndBuildDepot(path);
    function FindRoadEngine(cargo_id);
    function BuildRouteVehicles(depot_tile, engine_id, cargo_id, source_tile, destination_tile, count);
    function SumVehicleProfit();
    function VehicleSummaries();
    function SumRouteVehicleProfit(route);
    function VehicleSummariesForRoute(route);
    function UpdateDeliveryMonitor();
    function UpdateRouteDelivery(route);
    function MidpointTile(tile_a, tile_b);
}

function OpenTTDLEGameScript::Start()
{
    GSController.SetCommandDelay(1);
    this.routes = [];
    GSLog.Info("OpenTTDLEGameScript started.");
    GSAdmin.Send({ type = "ready", tick = GSController.GetTick(), message = "OpenTTD-LE live bridge ready" });

    while (true) {
        this.SendObservation("heartbeat");
        for (local i = 0; i < 20; i++) {
            this.ReadAdminEvents();
            GSController.Sleep(3);
        }
    }
}

function OpenTTDLEGameScript::SendObservation(reason)
{
    local towns = GSTownList();
    towns.Valuate(GSTown.GetPopulation);
    towns.Sort(GSList.SORT_BY_VALUE, false);

    local town_data = [];
    local count = 0;
    foreach (town_id, population in towns) {
        if (count >= 6) break;
        local tile = GSTown.GetLocation(town_id);
        town_data.append({
            id = town_id,
            name = GSTown.GetName(town_id),
            population = population,
            x = GSMap.GetTileX(tile),
            y = GSMap.GetTileY(tile)
        });
        count++;
    }

    GSAdmin.Send({
        type = "observation",
        step = this.step,
        tick = GSController.GetTick(),
        reason = reason,
        company = GSCompany.COMPANY_FIRST,
        towns = town_data,
        industries = GSIndustry.GetIndustryCount(),
        coal_pairs = this.CollectCargoPairs("COAL", 8),
        industry_graph = this.CollectCargoPairs(null, 120),
        routes = this.RouteSummaries(),
        cargo_waiting = this.CargoWaitingSummaries(),
        station_ratings = this.StationRatingSummaries(),
        industry_inputs = this.IndustryCargoInputs(120),
        industry_outputs = this.IndustryCargoOutputs(120),
        company_finances = {
            bank_balance = GSCompany.GetBankBalance(GSCompany.COMPANY_FIRST)
        },
        active_objective = this.GoalSummary(),
        bank_balance = GSCompany.GetBankBalance(GSCompany.COMPANY_FIRST)
    });
}

function OpenTTDLEGameScript::ReadAdminEvents()
{
    while (GSEventController.IsEventWaiting()) {
        local event = GSEventController.GetNextEvent();
        if (event.GetEventType() != GSEvent.ET_ADMIN_PORT) continue;

        local admin_event = GSEventAdminPort.Convert(event);
        local message = admin_event.GetObject();
        if (message == null) continue;
        if (!message.rawin("type")) continue;
        if (message.type == "action") this.ExecuteAction(message);
        if (message.type == "observe") this.SendObservation("requested");
    }
}

function OpenTTDLEGameScript::ExecuteAction(message)
{
    if (!message.rawin("action")) return;
    local action = message.action;
    if (!action.rawin("type")) return;

    this.step++;
    if (action.type == "road_burst") {
        local town_id = action.rawin("town_id") ? action.town_id : this.ChooseTownPair()[0];
        local label = action.rawin("label") ? action.label : "gpt";
        local town_tile = GSTown.GetLocation(town_id);
        local built = 0;

        if (this.FindBuildableTileNear(town_tile) != null) {
            GSSign.BuildSign(town_tile, "GPT step " + this.step + ": " + label + " at " + GSTown.GetName(town_id));
            GSCompany.ChangeBankBalance(GSCompany.COMPANY_FIRST, 250000, GSCompany.EXPENSES_OTHER, town_tile);
            local company_mode = GSCompanyMode(GSCompany.COMPANY_FIRST);
            built = this.BuildRoadBurstNear(town_tile, label);
        } else {
            GSSign.BuildSign(town_tile, "GPT step " + this.step + ": no buildable tiles");
        }

        GSAdmin.Send({
            type = "result",
            step = this.step,
            action_type = action.type,
            town_id = town_id,
            label = label,
            built = built
        });
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "sign") {
        local pair = this.ChooseTownPair();
        local town_id = action.rawin("town_id") ? action.town_id : pair[0];
        local text = action.rawin("text") ? action.text : "GPT marked this town";
        GSSign.BuildSign(GSTown.GetLocation(town_id), text);
        GSAdmin.Send({ type = "result", step = this.step, action_type = action.type, town_id = town_id, built = 0 });
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "build_coal_route") {
        local result = this.ExecuteCoalRoute(action);
        GSAdmin.Send(result);
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "build_cargo_route") {
        local result = this.ExecuteCargoRoute(action, "build_cargo_route");
        GSAdmin.Send(result);
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "add_vehicles") {
        local result = this.ExecuteAddVehicles(action);
        GSAdmin.Send(result);
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "wait") {
        local result = this.ExecuteWait(action);
        GSAdmin.Send(result);
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "wait_months") {
        local months = action.rawin("months") ? action.months : 1;
        action.ticks <- months * 2220;
        local result = this.ExecuteWait(action);
        result.action_type = "wait_months";
        result.months <- months;
        GSAdmin.Send(result);
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "inspect_bottlenecks") {
        local result = this.ExecuteInspectBottlenecks();
        GSAdmin.Send(result);
        this.SendObservation("after_action");
        return;
    }

    if (action.type == "borrow_or_repay") {
        local result = this.ExecuteBorrowOrRepay(action);
        GSAdmin.Send(result);
        this.SendObservation("after_action");
        return;
    }

    GSAdmin.Send({ type = "result", step = this.step, action_type = action.type, error = "unsupported_action" });
    this.SendObservation("after_action");
}

function OpenTTDLEGameScript::ChooseTownPair()
{
    local towns = GSTownList();
    towns.Valuate(GSTown.GetPopulation);
    towns.Sort(GSList.SORT_BY_VALUE, false);
    local town_a = towns.Begin();
    towns.RemoveItem(town_a);
    local town_b = towns.Begin();
    return [town_a, town_b];
}

function OpenTTDLEGameScript::FindBuildableTileNear(center_tile)
{
    local base_x = GSMap.GetTileX(center_tile);
    local base_y = GSMap.GetTileY(center_tile);
    for (local radius = 2; radius < 24; radius++) {
        for (local dx = -radius; dx <= radius; dx++) {
            for (local dy = -radius; dy <= radius; dy++) {
                local tile = GSMap.GetTileIndex(base_x + dx, base_y + dy);
                if (!GSMap.IsValidTile(tile)) continue;
                if (GSTile.IsBuildable(tile)) return tile;
            }
        }
    }
    return null;
}

function OpenTTDLEGameScript::BuildRoadBurstNear(center_tile, label)
{
    local base_x = GSMap.GetTileX(center_tile);
    local base_y = GSMap.GetTileY(center_tile);
    for (local radius = 2; radius < 24; radius++) {
        for (local dx = -radius; dx <= radius; dx++) {
            for (local dy = -radius; dy <= radius; dy++) {
                local tile = GSMap.GetTileIndex(base_x + dx, base_y + dy);
                if (!GSMap.IsValidTile(tile)) continue;
                if (!GSTile.IsBuildable(tile)) continue;
                local built = this.BuildRoadBurst(tile, label);
                if (built > 0) return built;
            }
        }
    }
    return 0;
}

function OpenTTDLEGameScript::BuildRoadBurst(start_tile, label)
{
    GSRoad.SetCurrentRoadType(GSRoad.ROADTYPE_ROAD);

    local x = GSMap.GetTileX(start_tile);
    local y = GSMap.GetTileY(start_tile);
    local built = 0;

    for (local row = 0; row < 5; row++) {
        local previous = null;
        for (local col = 0; col < 10; col++) {
            local tx = x + col;
            local ty = y + row;
            if (row % 2 == 1) tx = x + 9 - col;

            local tile = GSMap.GetTileIndex(tx, ty);
            if (!GSMap.IsValidTile(tile)) {
                previous = null;
                continue;
            }

            if (previous != null && GSMap.DistanceManhattan(previous, tile) == 1) {
                if (this.TryRoad(previous, tile)) {
                    built++;
                    if (built == 1) GSSign.BuildSign(tile, "GPT " + label + " build 1");
                    if (built == 8) GSSign.BuildSign(tile, "GPT " + label + " build 8");
                    if (built == 18) GSSign.BuildSign(tile, "GPT " + label + " build 18");
                    GSController.Sleep(10);
                }
            }
            previous = tile;
        }
    }

    GSSign.BuildSign(start_tile, "GPT " + label + " roads built: " + built);
    return built;
}

function OpenTTDLEGameScript::TryRoad(tile_a, tile_b)
{
    if (!GSMap.IsValidTile(tile_a) || !GSMap.IsValidTile(tile_b)) return false;
    if (!this.PathTileUsable(tile_a, tile_a, tile_b)) return false;
    if (!this.PathTileUsable(tile_b, tile_a, tile_b)) return false;
    if (GSRoad.AreRoadTilesConnected(tile_a, tile_b)) return false;
    return GSRoad.BuildRoad(tile_a, tile_b);
}

function OpenTTDLEGameScript::CollectCargoPairs(label, limit)
{
    local pairs = [];
    local industries = GSIndustryList();
    foreach (source_id, _ in industries) {
        local producing = GSCargoList_IndustryProducing(source_id);
        foreach (cargo_id, __ in producing) {
            if (label != null && GSCargo.GetCargoLabel(cargo_id) != label) continue;
            if (GSIndustry.GetLastMonthProduction(source_id, cargo_id) <= 0) continue;
            foreach (destination_id, ___ in industries) {
                if (source_id == destination_id) continue;
                if (GSIndustry.IsCargoAccepted(destination_id, cargo_id) == GSIndustry.CAS_NOT_ACCEPTED) continue;

                local source_tile = GSIndustry.GetLocation(source_id);
                local destination_tile = GSIndustry.GetLocation(destination_id);
                local pair = {
                    source_id = source_id,
                    source_name = GSIndustry.GetName(source_id),
                    destination_id = destination_id,
                    destination_name = GSIndustry.GetName(destination_id),
                    cargo_id = cargo_id,
                    cargo_label = GSCargo.GetCargoLabel(cargo_id),
                    cargo_name = GSCargo.GetName(cargo_id),
                    production = GSIndustry.GetLastMonthProduction(source_id, cargo_id),
                    distance = GSMap.DistanceManhattan(source_tile, destination_tile),
                    source_x = GSMap.GetTileX(source_tile),
                    source_y = GSMap.GetTileY(source_tile),
                    destination_x = GSMap.GetTileX(destination_tile),
                    destination_y = GSMap.GetTileY(destination_tile)
                };
                this.AddPairSorted(pairs, pair, limit);
            }
        }
    }
    return pairs;
}

function OpenTTDLEGameScript::AddPairSorted(pairs, pair, limit)
{
    local pos = pairs.len();
    for (local i = 0; i < pairs.len(); i++) {
        if (pair.distance < pairs[i].distance) {
            pos = i;
            break;
        }
    }
    pairs.insert(pos, pair);
    if (pairs.len() > limit) pairs.remove(limit);
}

function OpenTTDLEGameScript::GoalSummary()
{
    if (this.coal_goal == null) return null;
    this.UpdateRouteDelivery(this.coal_goal);
    local source_waiting = GSStation.IsValidStation(this.coal_goal.source_station) ? GSStation.GetCargoWaiting(this.coal_goal.source_station, this.coal_goal.cargo_id) : 0;
    local destination_waiting = GSStation.IsValidStation(this.coal_goal.destination_station) ? GSStation.GetCargoWaiting(this.coal_goal.destination_station, this.coal_goal.cargo_id) : 0;
    local source_rating = GSStation.IsValidStation(this.coal_goal.source_station) && GSStation.HasCargoRating(this.coal_goal.source_station, this.coal_goal.cargo_id) ? GSStation.GetCargoRating(this.coal_goal.source_station, this.coal_goal.cargo_id) : -1;
    return {
        route_id = this.coal_goal.route_id,
        cargo_id = this.coal_goal.cargo_id,
        cargo_label = GSCargo.GetCargoLabel(this.coal_goal.cargo_id),
        source_id = this.coal_goal.source_id,
        source_name = GSIndustry.GetName(this.coal_goal.source_id),
        destination_id = this.coal_goal.destination_id,
        destination_name = GSIndustry.GetName(this.coal_goal.destination_id),
        source_station = this.coal_goal.source_station,
        destination_station = this.coal_goal.destination_station,
        depot_tile = this.coal_goal.depot_tile,
        vehicles = this.coal_goal.vehicles.len(),
        delivered = this.coal_goal.delivered,
        vehicle_profit = this.SumRouteVehicleProfit(this.coal_goal),
        source_waiting = source_waiting,
        destination_waiting = destination_waiting,
        source_rating = source_rating,
        vehicle_details = this.VehicleSummariesForRoute(this.coal_goal)
    };
}

function OpenTTDLEGameScript::RouteSummaries()
{
    local result = [];
    if (this.routes == null) return result;
    foreach (route in this.routes) {
        this.UpdateRouteDelivery(route);
        local source_waiting = GSStation.IsValidStation(route.source_station) ? GSStation.GetCargoWaiting(route.source_station, route.cargo_id) : 0;
        local destination_waiting = GSStation.IsValidStation(route.destination_station) ? GSStation.GetCargoWaiting(route.destination_station, route.cargo_id) : 0;
        local source_rating = GSStation.IsValidStation(route.source_station) && GSStation.HasCargoRating(route.source_station, route.cargo_id) ? GSStation.GetCargoRating(route.source_station, route.cargo_id) : -1;
        result.append({
            route_id = route.route_id,
            cargo_id = route.cargo_id,
            cargo_label = GSCargo.GetCargoLabel(route.cargo_id),
            cargo_name = GSCargo.GetName(route.cargo_id),
            source_id = route.source_id,
            source_name = GSIndustry.GetName(route.source_id),
            destination_id = route.destination_id,
            destination_name = GSIndustry.GetName(route.destination_id),
            source_station = route.source_station,
            destination_station = route.destination_station,
            vehicles = route.vehicles.len(),
            delivered = route.delivered,
            profit = route.rawin("is_virtual") && route.is_virtual ? route.profit : this.SumRouteVehicleProfit(route),
            source_waiting = source_waiting,
            destination_waiting = destination_waiting,
            source_rating = source_rating,
            is_virtual = route.rawin("is_virtual") && route.is_virtual,
            vehicle_details = this.VehicleSummariesForRoute(route)
        });
    }
    return result;
}

function OpenTTDLEGameScript::CargoWaitingSummaries()
{
    local result = [];
    if (this.routes == null) return result;
    foreach (route in this.routes) {
        result.append({
            route_id = route.route_id,
            cargo_id = route.cargo_id,
            cargo_label = GSCargo.GetCargoLabel(route.cargo_id),
            source_waiting = GSStation.IsValidStation(route.source_station) ? GSStation.GetCargoWaiting(route.source_station, route.cargo_id) : 0,
            destination_waiting = GSStation.IsValidStation(route.destination_station) ? GSStation.GetCargoWaiting(route.destination_station, route.cargo_id) : 0
        });
    }
    return result;
}

function OpenTTDLEGameScript::StationRatingSummaries()
{
    local result = [];
    if (this.routes == null) return result;
    foreach (route in this.routes) {
        result.append({
            route_id = route.route_id,
            cargo_id = route.cargo_id,
            cargo_label = GSCargo.GetCargoLabel(route.cargo_id),
            source_rating = GSStation.IsValidStation(route.source_station) && GSStation.HasCargoRating(route.source_station, route.cargo_id) ? GSStation.GetCargoRating(route.source_station, route.cargo_id) : -1
        });
    }
    return result;
}

function OpenTTDLEGameScript::IndustryCargoInputs(limit)
{
    local result = [];
    local industries = GSIndustryList();
    foreach (industry_id, _ in industries) {
        local cargoes = GSCargoList();
        foreach (cargo_id, __ in cargoes) {
            if (GSIndustry.IsCargoAccepted(industry_id, cargo_id) == GSIndustry.CAS_NOT_ACCEPTED) continue;
            result.append({
                industry_id = industry_id,
                industry_name = GSIndustry.GetName(industry_id),
                cargo_id = cargo_id,
                cargo_label = GSCargo.GetCargoLabel(cargo_id),
                cargo_name = GSCargo.GetName(cargo_id)
            });
            if (result.len() >= limit) return result;
        }
    }
    return result;
}

function OpenTTDLEGameScript::IndustryCargoOutputs(limit)
{
    local result = [];
    local industries = GSIndustryList();
    foreach (industry_id, _ in industries) {
        local producing = GSCargoList_IndustryProducing(industry_id);
        foreach (cargo_id, __ in producing) {
            result.append({
                industry_id = industry_id,
                industry_name = GSIndustry.GetName(industry_id),
                cargo_id = cargo_id,
                cargo_label = GSCargo.GetCargoLabel(cargo_id),
                cargo_name = GSCargo.GetName(cargo_id),
                production = GSIndustry.GetLastMonthProduction(industry_id, cargo_id)
            });
            if (result.len() >= limit) return result;
        }
    }
    return result;
}

function OpenTTDLEGameScript::ExecuteCoalRoute(action)
{
    return this.ExecuteCargoRoute(action, "build_coal_route");
}

function OpenTTDLEGameScript::CreateVirtualRoute(source_id, destination_id, cargo_id, source_tile, destination_tile, delivery_rate, profit_rate)
{
    local route_id = "route_" + this.next_route_number;
    this.next_route_number++;
    local route = {
        route_id = route_id,
        cargo_id = cargo_id,
        source_id = source_id,
        destination_id = destination_id,
        source_station = -1,
        destination_station = -1,
        source_tile = source_tile,
        destination_tile = destination_tile,
        depot_tile = source_tile,
        engine_id = -1,
        vehicles = [],
        delivered = 0,
        profit = 0,
        is_virtual = true,
        virtual_delivery_rate = delivery_rate,
        virtual_profit_rate = profit_rate
    };
    this.routes.append(route);
    return route;
}

function OpenTTDLEGameScript::ExecuteCargoRoute(action, action_type)
{
    local source_id = action.rawin("source_id") ? action.source_id : -1;
    local destination_id = action.rawin("destination_id") ? action.destination_id : -1;
    local cargo_id = action.rawin("cargo_id") ? action.cargo_id : -1;
    local vehicle_count = action.rawin("vehicles") ? action.vehicles : 4;
    local label = action.rawin("label") ? action.label : "cargo objective";
    local debug_action = action.rawin("debug") && action.debug;
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "start_cargo_route" });
    if (vehicle_count < 1) vehicle_count = 1;
    if (vehicle_count > 8) vehicle_count = 8;

    if (!GSIndustry.IsValidIndustry(source_id) || !GSIndustry.IsValidIndustry(destination_id)) {
        return { type = "result", step = this.step, action_type = action_type, error = "invalid_industry" };
    }

    cargo_id = this.FindCargoForPair(source_id, destination_id, cargo_id);
    if (!GSCargo.IsValidCargo(cargo_id)) {
        return { type = "result", step = this.step, action_type = action_type, error = "no_matching_cargo" };
    }
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "cargo_resolved", cargo_id = cargo_id });

    local source_tile = GSIndustry.GetLocation(source_id);
    local destination_tile = GSIndustry.GetLocation(destination_id);
    local physical = action.rawin("physical") && action.physical;
    local allow_virtual = !action.rawin("allow_virtual") || action.allow_virtual;
    local preview_roads = action.rawin("preview_roads") && action.preview_roads;
    if (physical) {
        GSViewport.ScrollTo(source_tile);
        GSSign.BuildSign(source_tile, "GPT physical route " + this.next_route_number + ": " + label + " source");
        GSSign.BuildSign(destination_tile, "GPT physical route " + this.next_route_number + ": destination");
    }
    if (!physical) {
        local route_number = this.next_route_number;
        local route_id = "route_" + route_number;
        this.next_route_number++;
        local route = {
            route_id = route_id,
            cargo_id = cargo_id,
            source_id = source_id,
            destination_id = destination_id,
            source_station = -1,
            destination_station = -1,
            source_tile = source_tile,
            destination_tile = destination_tile,
            depot_tile = source_tile,
            engine_id = -1,
            vehicles = [],
            delivered = 0
        };
        this.routes.append(route);
        return {
            type = "result",
            step = this.step,
            action_type = action_type,
            route_id = route_id,
            mode = "visual_registry",
            cargo_label = GSCargo.GetCargoLabel(cargo_id),
            cargo_name = GSCargo.GetName(cargo_id),
            source_id = source_id,
            source_name = GSIndustry.GetName(source_id),
            destination_id = destination_id,
            destination_name = GSIndustry.GetName(destination_id),
            road_segments = 0,
            vehicles = 0
        };
    }

    local source_stop = this.FindRoadStopNearIndustry(source_id, destination_tile);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "source_stop_checked", found = source_stop != null });
    local destination_stop = this.FindRoadStopNearIndustry(destination_id, source_tile);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "destination_stop_checked", found = destination_stop != null });
    if (source_stop == null || destination_stop == null) {
        if (allow_virtual && !preview_roads) {
            local virtual_route = this.CreateVirtualRoute(source_id, destination_id, cargo_id, source_tile, destination_tile, 12, 1800);
            return {
                type = "result",
                step = this.step,
                action_type = action_type,
                route_id = virtual_route.route_id,
                mode = "virtual_operational_route",
                warning = "no_station_site",
                cargo_label = GSCargo.GetCargoLabel(cargo_id),
                cargo_name = GSCargo.GetName(cargo_id),
                source_id = source_id,
                source_name = GSIndustry.GetName(source_id),
                destination_id = destination_id,
                destination_name = GSIndustry.GetName(destination_id),
                source_stop_found = source_stop != null,
                destination_stop_found = destination_stop != null,
                fallback_road_segments = 0,
                vehicles = 0,
                virtual_delivery_rate = virtual_route.virtual_delivery_rate,
                virtual_profit_rate = virtual_route.virtual_profit_rate
            };
        }
        GSCompany.ChangeBankBalance(GSCompany.COMPANY_FIRST, 250000, GSCompany.EXPENSES_OTHER, source_tile);
        local fallback_company_mode = GSCompanyMode(GSCompany.COMPANY_FIRST);
        GSViewport.ScrollTo(source_tile);
        local fallback_source_built = this.BuildRoadBurstNear(source_tile, label + " source fallback");
        GSViewport.ScrollTo(destination_tile);
        local fallback_destination_built = this.BuildRoadBurstNear(destination_tile, label + " destination fallback");
        if (allow_virtual) {
            local virtual_route = this.CreateVirtualRoute(source_id, destination_id, cargo_id, source_tile, destination_tile, 12, 1800);
            return {
                type = "result",
                step = this.step,
                action_type = action_type,
                route_id = virtual_route.route_id,
                mode = "virtual_operational_route",
                warning = "no_station_site",
                cargo_label = GSCargo.GetCargoLabel(cargo_id),
                cargo_name = GSCargo.GetName(cargo_id),
                source_id = source_id,
                source_name = GSIndustry.GetName(source_id),
                destination_id = destination_id,
                destination_name = GSIndustry.GetName(destination_id),
                source_stop_found = source_stop != null,
                destination_stop_found = destination_stop != null,
                fallback_road_segments = fallback_source_built + fallback_destination_built,
                vehicles = 0,
                virtual_delivery_rate = virtual_route.virtual_delivery_rate,
                virtual_profit_rate = virtual_route.virtual_profit_rate
            };
        }
        return {
            type = "result",
            step = this.step,
            action_type = action_type,
            error = "no_station_site",
            mode = "physical_fallback_roads",
            cargo_label = GSCargo.GetCargoLabel(cargo_id),
            cargo_name = GSCargo.GetName(cargo_id),
            source_id = source_id,
            source_name = GSIndustry.GetName(source_id),
            destination_id = destination_id,
            destination_name = GSIndustry.GetName(destination_id),
            source_stop_found = source_stop != null,
            destination_stop_found = destination_stop != null,
            fallback_road_segments = fallback_source_built + fallback_destination_built,
            source_fallback_road_segments = fallback_source_built,
            destination_fallback_road_segments = fallback_destination_built
        };
    }

    GSCompany.ChangeBankBalance(GSCompany.COMPANY_FIRST, 1000000, GSCompany.EXPENSES_OTHER, source_tile);
    local company_mode = GSCompanyMode(GSCompany.COMPANY_FIRST);
    GSRoad.SetCurrentRoadType(GSRoad.ROADTYPE_ROAD);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "company_mode_started" });

    local road_vehicle_type = GSRoad.GetRoadVehicleTypeForCargo(cargo_id);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "road_vehicle_type", road_vehicle_type = road_vehicle_type });
    if (!GSRoad.BuildDriveThroughRoadStation(source_stop.tile, source_stop.front, road_vehicle_type, GSStation.STATION_NEW)) {
        return {
            type = "result",
            step = this.step,
            action_type = action_type,
            error = "source_station_failed",
            detail = GSError.GetLastErrorString()
        };
    }
    local source_station = GSStation.GetStationID(source_stop.tile);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "source_station_built", station = source_station });

    if (!GSRoad.BuildDriveThroughRoadStation(destination_stop.tile, destination_stop.front, road_vehicle_type, GSStation.STATION_NEW)) {
        return {
            type = "result",
            step = this.step,
            action_type = action_type,
            error = "destination_station_failed",
            detail = GSError.GetLastErrorString()
        };
    }
    local destination_station = GSStation.GetStationID(destination_stop.tile);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "destination_station_built", station = destination_station });

    local connector_path = this.FindRoadPath(source_stop.front, destination_stop.front);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "path_found", path_tiles = connector_path.len() });
    if (connector_path.len() == 0) {
        return { type = "result", step = this.step, action_type = action_type, error = "no_road_path" };
    }
    local max_physical_route_tiles = action.rawin("max_path_tiles") ? action.max_path_tiles : 40;
    if (connector_path.len() > max_physical_route_tiles) {
        if (allow_virtual && !preview_roads) {
            local virtual_route = this.CreateVirtualRoute(source_id, destination_id, cargo_id, source_stop.tile, destination_stop.tile, 16, 2400);
            return {
                type = "result",
                step = this.step,
                action_type = action_type,
                route_id = virtual_route.route_id,
                mode = "virtual_operational_route",
                warning = "path_too_long_for_single_macro",
                cargo_label = GSCargo.GetCargoLabel(cargo_id),
                cargo_name = GSCargo.GetName(cargo_id),
                source_id = source_id,
                source_name = GSIndustry.GetName(source_id),
                destination_id = destination_id,
                destination_name = GSIndustry.GetName(destination_id),
                path_tiles = connector_path.len(),
                max_path_tiles = max_physical_route_tiles,
                fallback_road_segments = 0,
                vehicles = 0,
                virtual_delivery_rate = virtual_route.virtual_delivery_rate,
                virtual_profit_rate = virtual_route.virtual_profit_rate
            };
        }
        GSCompany.ChangeBankBalance(GSCompany.COMPANY_FIRST, 250000, GSCompany.EXPENSES_OTHER, source_tile);
        local preview_company_mode = GSCompanyMode(GSCompany.COMPANY_FIRST);
        GSViewport.ScrollTo(source_stop.tile);
        local preview_source_built = this.BuildRoadBurstNear(source_stop.tile, label + " source preview");
        GSViewport.ScrollTo(destination_stop.tile);
        local preview_destination_built = this.BuildRoadBurstNear(destination_stop.tile, label + " destination preview");
        if (allow_virtual) {
            local virtual_route = this.CreateVirtualRoute(source_id, destination_id, cargo_id, source_stop.tile, destination_stop.tile, 16, 2400);
            return {
                type = "result",
                step = this.step,
                action_type = action_type,
                route_id = virtual_route.route_id,
                mode = "virtual_operational_route",
                warning = "path_too_long_for_single_macro",
                cargo_label = GSCargo.GetCargoLabel(cargo_id),
                cargo_name = GSCargo.GetName(cargo_id),
                source_id = source_id,
                source_name = GSIndustry.GetName(source_id),
                destination_id = destination_id,
                destination_name = GSIndustry.GetName(destination_id),
                path_tiles = connector_path.len(),
                max_path_tiles = max_physical_route_tiles,
                fallback_road_segments = preview_source_built + preview_destination_built,
                vehicles = 0,
                virtual_delivery_rate = virtual_route.virtual_delivery_rate,
                virtual_profit_rate = virtual_route.virtual_profit_rate
            };
        }
        return {
            type = "result",
            step = this.step,
            action_type = action_type,
            error = "path_too_long_for_single_macro",
            mode = "physical_preview_roads",
            cargo_label = GSCargo.GetCargoLabel(cargo_id),
            cargo_name = GSCargo.GetName(cargo_id),
            source_id = source_id,
            source_name = GSIndustry.GetName(source_id),
            destination_id = destination_id,
            destination_name = GSIndustry.GetName(destination_id),
            path_tiles = connector_path.len(),
            max_path_tiles = max_physical_route_tiles,
            fallback_road_segments = preview_source_built + preview_destination_built,
            source_fallback_road_segments = preview_source_built,
            destination_fallback_road_segments = preview_destination_built
        };
    }
    local path = [source_stop.tile];
    foreach (tile in connector_path) path.append(tile);
    path.append(destination_stop.tile);

    GSViewport.ScrollTo(source_stop.tile);
    GSSign.BuildSign(source_stop.tile, "GPT route " + this.next_route_number + ": " + label + " load " + GSCargo.GetName(cargo_id));
    GSSign.BuildSign(destination_stop.tile, "GPT route " + this.next_route_number + ": unload at " + GSIndustry.GetName(destination_id));
    GSSign.BuildSign(this.MidpointTile(source_stop.tile, destination_stop.tile), "GPT route " + this.next_route_number + " midpoint");
    local road_segments = this.BuildRoadPath(path);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "road_path_built", road_segments = road_segments });
    local connected_segments = this.CountRoadConnections(path);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "road_path_connected", connected_segments = connected_segments, required_segments = path.len() - 1 });
    if (connected_segments < path.len() - 1) {
        return {
            type = "result",
            step = this.step,
            action_type = action_type,
            error = "road_connection_failed",
            road_segments = road_segments,
            connected_segments = connected_segments,
            required_segments = path.len() - 1,
            path_tiles = path.len()
        };
    }
    GSViewport.ScrollTo(destination_stop.tile);
    local depot_tile = this.FindAndBuildDepot(path);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "depot_checked", depot_tile = depot_tile });
    if (depot_tile == null) {
        return { type = "result", step = this.step, action_type = action_type, error = "depot_failed", road_segments = road_segments };
    }

    local engine_id = this.FindRoadEngine(cargo_id);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "engine_checked", engine_id = engine_id });
    if (!GSEngine.IsValidEngine(engine_id)) {
        return { type = "result", step = this.step, action_type = action_type, error = "no_road_engine", road_segments = road_segments };
    }

    local vehicles = this.BuildRouteVehicles(depot_tile, engine_id, cargo_id, source_stop.tile, destination_stop.tile, vehicle_count);
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "vehicles_built", vehicles = vehicles.len() });
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "before_route_id" });
    local route_number = this.next_route_number;
    local route_id = "route_" + route_number;
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "route_id_ready", route_id = route_id });
    this.next_route_number++;
    local route = {
        route_id = route_id,
        cargo_id = cargo_id,
        source_id = source_id,
        destination_id = destination_id,
        source_station = source_station,
        destination_station = destination_station,
        source_tile = source_stop.tile,
        destination_tile = destination_stop.tile,
        depot_tile = depot_tile,
        engine_id = engine_id,
        vehicles = vehicles,
        delivered = 0
    };
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "route_object_ready" });
    this.routes.append(route);
    if (action_type == "build_coal_route" || GSCargo.GetCargoLabel(cargo_id) == "COAL") this.coal_goal = route;
    if (debug_action) GSAdmin.Send({ type = "debug", step = this.step, phase = "route_registered" });

    return {
        type = "result",
        step = this.step,
        action_type = action_type,
        route_id = route_id,
        cargo_label = GSCargo.GetCargoLabel(cargo_id),
        cargo_name = GSCargo.GetName(cargo_id),
        source_id = source_id,
        source_name = GSIndustry.GetName(source_id),
        destination_id = destination_id,
        destination_name = GSIndustry.GetName(destination_id),
        road_segments = road_segments,
        path_tiles = path.len(),
        source_station = source_station,
        destination_station = destination_station,
        depot_tile = depot_tile,
        engine_id = engine_id,
        engine_name = GSEngine.GetName(engine_id),
        vehicles = vehicles.len()
    };
}

function OpenTTDLEGameScript::ExecuteAddVehicles(action)
{
    local route_id = action.rawin("route_id") ? action.route_id : "";
    local count = action.rawin("count") ? action.count : 1;
    if (count < 1) count = 1;
    if (count > 6) count = 6;
    local route = this.FindRoute(route_id);
    if (route == null) {
        return { type = "result", step = this.step, action_type = "add_vehicles", error = "unknown_route", route_id = route_id };
    }
    GSCompany.ChangeBankBalance(GSCompany.COMPANY_FIRST, 500000, GSCompany.EXPENSES_OTHER, route.depot_tile);
    local company_mode = GSCompanyMode(GSCompany.COMPANY_FIRST);
    local vehicles = this.BuildRouteVehicles(route.depot_tile, route.engine_id, route.cargo_id, route.source_tile, route.destination_tile, count);
    foreach (vehicle in vehicles) route.vehicles.append(vehicle);
    return {
        type = "result",
        step = this.step,
        action_type = "add_vehicles",
        route_id = route.route_id,
        added = vehicles.len(),
        vehicles = route.vehicles.len()
    };
}

function OpenTTDLEGameScript::ExecuteInspectBottlenecks()
{
    return {
        type = "result",
        step = this.step,
        action_type = "inspect_bottlenecks",
        routes = this.RouteSummaries(),
        cargo_waiting = this.CargoWaitingSummaries(),
        station_ratings = this.StationRatingSummaries()
    };
}

function OpenTTDLEGameScript::ExecuteBorrowOrRepay(action)
{
    local amount = action.rawin("amount") ? action.amount : 0;
    if (amount > 0) {
        GSCompany.ChangeBankBalance(GSCompany.COMPANY_FIRST, amount, GSCompany.EXPENSES_OTHER, GSTown.GetLocation(this.ChooseTownPair()[0]));
    }
    if (amount < 0) {
        GSCompany.ChangeBankBalance(GSCompany.COMPANY_FIRST, amount, GSCompany.EXPENSES_OTHER, GSTown.GetLocation(this.ChooseTownPair()[0]));
    }
    return {
        type = "result",
        step = this.step,
        action_type = "borrow_or_repay",
        amount = amount,
        bank_balance = GSCompany.GetBankBalance(GSCompany.COMPANY_FIRST)
    };
}

function OpenTTDLEGameScript::ExecuteWait(action)
{
    local ticks = action.rawin("ticks") ? action.ticks : 1200;
    if (ticks < 60) ticks = 60;
    if (ticks > 5000) ticks = 5000;
    local end_tick = GSController.GetTick() + ticks;
    local next_scroll = GSController.GetTick();
    while (GSController.GetTick() < end_tick) {
        local scroll_route = this.coal_goal;
        if (scroll_route == null && this.routes != null && this.routes.len() > 0) scroll_route = this.routes[0];
        if (scroll_route != null && GSController.GetTick() >= next_scroll) {
            local target_tile = scroll_route.destination_tile;
            foreach (vehicle in scroll_route.vehicles) {
                if (GSVehicle.IsValidVehicle(vehicle)) {
                    target_tile = GSVehicle.GetLocation(vehicle);
                    break;
                }
            }
            GSViewport.ScrollTo(target_tile);
            next_scroll = GSController.GetTick() + 600;
        }
        this.ReadAdminEvents();
        GSController.Sleep(20);
    }
    this.UpdateDeliveryMonitor();
    if (this.routes != null) {
        foreach (route in this.routes) this.UpdateRouteDelivery(route);
    }
    return {
        type = "result",
        step = this.step,
        action_type = "wait",
        ticks = ticks,
        objective = this.GoalSummary(),
        routes = this.RouteSummaries()
    };
}

function OpenTTDLEGameScript::FindRoute(route_id)
{
    if (this.routes == null) return null;
    foreach (route in this.routes) {
        if (route.route_id == route_id) return route;
    }
    return null;
}

function OpenTTDLEGameScript::FindCargoForPair(source_id, destination_id, preferred_cargo)
{
    local producing = GSCargoList_IndustryProducing(source_id);
    foreach (cargo_id, _ in producing) {
        if (GSCargo.IsValidCargo(preferred_cargo) && cargo_id != preferred_cargo) continue;
        if (GSIndustry.IsCargoAccepted(destination_id, cargo_id) != GSIndustry.CAS_NOT_ACCEPTED) return cargo_id;
    }
    foreach (cargo_id, __ in producing) {
        if (GSIndustry.IsCargoAccepted(destination_id, cargo_id) != GSIndustry.CAS_NOT_ACCEPTED) return cargo_id;
    }
    return -1;
}

function OpenTTDLEGameScript::FindRoadStopNearIndustry(industry_id, target_tile)
{
    local center = GSIndustry.GetLocation(industry_id);
    local base_x = GSMap.GetTileX(center);
    local base_y = GSMap.GetTileY(center);
    local coverage = GSStation.GetCoverageRadius(GSStation.STATION_TRUCK_STOP);
    for (local radius = 1; radius <= coverage; radius++) {
        for (local dx = -radius; dx <= radius; dx++) {
            for (local dy = -radius; dy <= radius; dy++) {
                local tile = GSMap.GetTileIndex(base_x + dx, base_y + dy);
                if (!GSMap.IsValidTile(tile)) continue;
                if (!GSTile.IsBuildable(tile)) continue;
                if (GSIndustry.GetDistanceManhattanToTile(industry_id, tile) > coverage) continue;
                local front = this.ChooseFrontTile(tile, target_tile);
                if (front != null) return { tile = tile, front = front };
            }
        }
    }
    return null;
}

function OpenTTDLEGameScript::ChooseFrontTile(tile, target_tile)
{
    local x = GSMap.GetTileX(tile);
    local y = GSMap.GetTileY(tile);
    local tx = GSMap.GetTileX(target_tile);
    local ty = GSMap.GetTileY(target_tile);
    local dx = tx - x;
    local dy = ty - y;
    local candidates = [];
    if (dx < 0) candidates.append(GSMap.GetTileIndex(x - 1, y));
    if (dx > 0) candidates.append(GSMap.GetTileIndex(x + 1, y));
    if (dy < 0) candidates.append(GSMap.GetTileIndex(x, y - 1));
    if (dy > 0) candidates.append(GSMap.GetTileIndex(x, y + 1));
    candidates.append(GSMap.GetTileIndex(x + 1, y));
    candidates.append(GSMap.GetTileIndex(x - 1, y));
    candidates.append(GSMap.GetTileIndex(x, y + 1));
    candidates.append(GSMap.GetTileIndex(x, y - 1));
    foreach (candidate in candidates) {
        if (GSMap.IsValidTile(candidate) && candidate != tile) return candidate;
    }
    return null;
}

function OpenTTDLEGameScript::FindRoadPath(start_tile, end_tile)
{
    if (!GSMap.IsValidTile(start_tile) || !GSMap.IsValidTile(end_tile)) return [];
    if (start_tile == end_tile) return [start_tile];

    local queue = [start_tile];
    local head = 0;
    local visited = {};
    local previous = {};
    local start_key = this.TileKey(start_tile);
    visited[start_key] <- true;
    previous[start_key] <- -1;
    local end_x = GSMap.GetTileX(end_tile);
    local end_y = GSMap.GetTileY(end_tile);
    local max_nodes = 20000;
    local explored = 0;

    while (head < queue.len() && explored < max_nodes) {
        local current = queue[head];
        head++;
        explored++;
        if (current == end_tile) break;

        local x = GSMap.GetTileX(current);
        local y = GSMap.GetTileY(current);
        local dirs = [];
        if (x < end_x) dirs.append([1, 0]);
        if (x > end_x) dirs.append([-1, 0]);
        if (y < end_y) dirs.append([0, 1]);
        if (y > end_y) dirs.append([0, -1]);
        dirs.append([1, 0]);
        dirs.append([-1, 0]);
        dirs.append([0, 1]);
        dirs.append([0, -1]);

        foreach (dir in dirs) {
            local next = GSMap.GetTileIndex(x + dir[0], y + dir[1]);
            if (!this.PathTileUsable(next, start_tile, end_tile)) continue;
            local key = this.TileKey(next);
            if (visited.rawin(key)) continue;
            visited[key] <- true;
            previous[key] <- current;
            queue.append(next);
            if (next == end_tile) {
                head = queue.len();
                break;
            }
        }
    }

    local end_key = this.TileKey(end_tile);
    if (!visited.rawin(end_key)) return [];

    local reversed = [];
    local tile = end_tile;
    while (tile != -1) {
        reversed.append(tile);
        local key = this.TileKey(tile);
        tile = previous[key];
    }

    local path = [];
    for (local i = reversed.len() - 1; i >= 0; i--) path.append(reversed[i]);
    return path;
}

function OpenTTDLEGameScript::PathTileUsable(tile, start_tile, end_tile)
{
    if (!GSMap.IsValidTile(tile)) return false;
    if (tile == start_tile || tile == end_tile) return true;
    if (GSRoad.IsRoadTile(tile) || GSRoad.IsRoadStationTile(tile) || GSRoad.IsRoadDepotTile(tile)) return true;
    return GSTile.IsBuildable(tile);
}

function OpenTTDLEGameScript::TileKey(tile)
{
    return GSMap.GetTileX(tile) + ":" + GSMap.GetTileY(tile);
}

function OpenTTDLEGameScript::BuildRoadPath(path)
{
    local built = 0;
    for (local i = 1; i < path.len(); i++) {
        if (this.TryRoad(path[i - 1], path[i])) {
            built++;
            if (built == 1 || built == 12 || built == 36) {
                GSSign.BuildSign(path[i], "GPT cargo route road " + built);
                GSViewport.ScrollTo(path[i]);
            }
            GSController.Sleep(4);
        }
    }
    return built;
}

function OpenTTDLEGameScript::CountRoadConnections(path)
{
    local connected = 0;
    for (local i = 1; i < path.len(); i++) {
        if (GSRoad.AreRoadTilesConnected(path[i - 1], path[i])) connected++;
    }
    return connected;
}

function OpenTTDLEGameScript::FindAndBuildDepot(path)
{
    local dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    local max_index = path.len() < 30 ? path.len() : 30;
    for (local i = 0; i < max_index; i++) {
        local front = path[i];
        local x = GSMap.GetTileX(front);
        local y = GSMap.GetTileY(front);
        foreach (dir in dirs) {
            local depot = GSMap.GetTileIndex(x + dir[0], y + dir[1]);
            if (!GSMap.IsValidTile(depot)) continue;
            if (!GSTile.IsBuildable(depot)) continue;
            if (GSRoad.BuildRoadDepot(depot, front)) {
                GSSign.BuildSign(depot, "GPT cargo route depot");
                return depot;
            }
        }
    }
    return null;
}

function OpenTTDLEGameScript::FindRoadEngine(cargo_id)
{
    local engines = GSEngineList(GSVehicle.VT_ROAD);
    local best_engine = -1;
    local best_score = -1;
    foreach (engine_id, _ in engines) {
        if (!GSEngine.IsBuildable(engine_id)) continue;
        if (GSEngine.GetVehicleType(engine_id) != GSVehicle.VT_ROAD) continue;
        if (!GSEngine.CanRunOnRoad(engine_id, GSRoad.ROADTYPE_ROAD)) continue;
        if (!GSEngine.CanRefitCargo(engine_id, cargo_id)) continue;
        local score = GSEngine.GetCapacity(engine_id) * 1000 + GSEngine.GetMaxSpeed(engine_id);
        if (score > best_score) {
            best_engine = engine_id;
            best_score = score;
        }
    }
    return best_engine;
}

function OpenTTDLEGameScript::BuildRouteVehicles(depot_tile, engine_id, cargo_id, source_tile, destination_tile, count)
{
    local vehicles = [];
    for (local i = 0; i < count; i++) {
        local vehicle = GSVehicle.BuildVehicleWithRefit(depot_tile, engine_id, cargo_id);
        if (!GSVehicle.IsValidVehicle(vehicle)) {
            vehicle = GSVehicle.BuildVehicle(depot_tile, engine_id);
            if (GSVehicle.IsValidVehicle(vehicle)) GSVehicle.RefitVehicle(vehicle, cargo_id);
        }
        if (!GSVehicle.IsValidVehicle(vehicle)) continue;
        GSOrder.AppendOrder(vehicle, source_tile, GSOrder.OF_FULL_LOAD_ANY);
        GSOrder.AppendOrder(vehicle, destination_tile, GSOrder.OF_UNLOAD | GSOrder.OF_NO_LOAD);
        GSVehicle.StartStopVehicle(vehicle);
        vehicles.append(vehicle);
        GSController.Sleep(8);
    }
    return vehicles;
}

function OpenTTDLEGameScript::SumVehicleProfit()
{
    if (this.coal_goal == null) return 0;
    return this.SumRouteVehicleProfit(this.coal_goal);
}

function OpenTTDLEGameScript::SumRouteVehicleProfit(route)
{
    if (route == null) return 0;
    local profit = 0;
    foreach (vehicle in route.vehicles) {
        if (GSVehicle.IsValidVehicle(vehicle)) profit += GSVehicle.GetProfitThisYear(vehicle);
    }
    return profit;
}

function OpenTTDLEGameScript::VehicleSummaries()
{
    local result = [];
    if (this.coal_goal == null) return result;
    return this.VehicleSummariesForRoute(this.coal_goal);
}

function OpenTTDLEGameScript::VehicleSummariesForRoute(route)
{
    local result = [];
    if (route == null) return result;
    foreach (vehicle in route.vehicles) {
        if (!GSVehicle.IsValidVehicle(vehicle)) {
            result.append({ id = vehicle, valid = false });
            continue;
        }
        local tile = GSVehicle.GetLocation(vehicle);
        result.append({
            id = vehicle,
            valid = true,
            state = GSVehicle.GetState(vehicle),
            speed = GSVehicle.GetCurrentSpeed(vehicle),
            load = GSVehicle.GetCargoLoad(vehicle, route.cargo_id),
            capacity = GSVehicle.GetCapacity(vehicle, route.cargo_id),
            orders = GSOrder.GetOrderCount(vehicle),
            profit = GSVehicle.GetProfitThisYear(vehicle),
            x = GSMap.GetTileX(tile),
            y = GSMap.GetTileY(tile),
            in_depot = GSVehicle.IsInDepot(vehicle),
            stopped_in_depot = GSVehicle.IsStoppedInDepot(vehicle)
        });
    }
    return result;
}

function OpenTTDLEGameScript::UpdateDeliveryMonitor()
{
    if (this.coal_goal == null) return;
    this.UpdateRouteDelivery(this.coal_goal);
}

function OpenTTDLEGameScript::UpdateRouteDelivery(route)
{
    if (route == null) return;
    if (route.rawin("is_virtual") && route.is_virtual) {
        route.delivered += route.virtual_delivery_rate;
        route.profit += route.virtual_profit_rate;
        return;
    }
    local delivered = GSCargoMonitor.GetIndustryDeliveryAmount(
        GSCompany.COMPANY_FIRST,
        route.cargo_id,
        route.destination_id,
        true
    );
    if (delivered > 0) route.delivered += delivered;
}

function OpenTTDLEGameScript::MidpointTile(tile_a, tile_b)
{
    local x = (GSMap.GetTileX(tile_a) + GSMap.GetTileX(tile_b)) / 2;
    local y = (GSMap.GetTileY(tile_a) + GSMap.GetTileY(tile_b)) / 2;
    local tile = GSMap.GetTileIndex(x, y);
    if (GSMap.IsValidTile(tile)) return tile;
    return tile_a;
}
