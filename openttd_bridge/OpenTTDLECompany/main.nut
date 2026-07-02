class OpenTTDLECompany extends AIController {
    function Start();
}

function OpenTTDLECompany::Start()
{
    this.Sleep(1);
    AICompany.SetName("TycoonLE OpenTTD Company");
    AILog.Info("OpenTTDLECompany created a controlled company.");
    while (true) {
        this.Sleep(150);
    }
}
